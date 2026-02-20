import torch

from .lejepa_losses import EppsPulley, SlicingUnivariateTest


def _supports_3d_input(loss_module):
    # EppsPulley uses relative indexing, supports 3D (V, N, D)
    if isinstance(loss_module, SlicingUnivariateTest):
        return isinstance(loss_module.univariate_test, EppsPulley)
    return isinstance(loss_module, EppsPulley)


def _get_views_list(batch):
    if isinstance(batch, list):
        return batch
    elif isinstance(batch, dict) and "image" not in batch:
        views = [v for v in batch.values() if isinstance(v, dict) and "image" in v]
        return views if views else None
    return None


def _extract_embedding(embedding):
    # Extract CLS token from TIMM ViT output
    if embedding.dim() == 3:
        return embedding[:, 0, :]  # CLS token
    return embedding  # Already 2D (pooled)


def lejepa_forward(self, batch, stage):
    out = {}
    views = _get_views_list(batch)

    if views is not None:
        # Multi-view training
        V, N = len(views), views[0]["image"].size(0)

        # Single forward pass for all views
        all_images = torch.cat([view["image"] for view in views], dim=0)
        all_emb = _extract_embedding(self.backbone.forward_features(all_images))
        out["embedding"] = all_emb

        if "label" in views[0]:
            out["label"] = torch.cat([view["label"] for view in views], dim=0)

        if self.training:
            all_proj = self.projector(all_emb)
            proj_stacked = all_proj.reshape(V, N, -1)  # (V, N, proj_dim)

            # Invariance loss: compare each view to mean
            view_mean = proj_stacked.mean(0)
            inv_loss = (view_mean - proj_stacked).square().mean()

            # SIGReg loss
            if _supports_3d_input(self.sigreg_loss):
                sigreg_loss = self.sigreg_loss(proj_stacked)
            else:
                sigreg_loss = self.sigreg_loss(
                    proj_stacked.reshape(-1, proj_stacked.size(-1))
                )

            lamb = getattr(self, "lamb", 0.02)
            lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
            out["loss"] = lejepa_loss

            self.log(
                f"{stage}/sigreg",
                sigreg_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"{stage}/inv", inv_loss, on_step=True, on_epoch=True, sync_dist=True
            )
            self.log(
                f"{stage}/lejepa",
                lejepa_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        # Single-view (validation)
        out["embedding"] = _extract_embedding(self.backbone.forward_features(batch["image"]))
        if "label" in batch:
            out["label"] = batch["label"]

    return out
