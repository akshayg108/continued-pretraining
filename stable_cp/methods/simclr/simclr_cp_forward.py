import torch


def _extract_embedding(backbone_output, pool_strategy="cls"):
    if backbone_output.ndim == 3:
        if pool_strategy == "mean":
            return backbone_output[:, 1:, :].mean(dim=1)
        return backbone_output[:, 0, :]  # CLS token
    return backbone_output


def _get_views_list(batch):
    if isinstance(batch, list):
        return batch
    elif isinstance(batch, dict) and "image" not in batch:
        views = [v for v in batch.values() if isinstance(v, dict) and "image" in v]
        return views if views else None
    return None


def simclr_cp_forward(self, batch, stage):
    out = {}
    views = _get_views_list(batch)
    pool_strategy = getattr(self, "pool_strategy", "cls")

    if views is not None:
        if len(views) != 2:
            raise ValueError(f"SimCLR requires 2 views, got {len(views)}")

        embeddings = [
            _extract_embedding(self.backbone.forward_features(v["image"]), pool_strategy)
            for v in views
        ]
        out["embedding"] = torch.cat(embeddings, dim=0)

        if "label" in views[0]:
            out["label"] = torch.cat([v["label"] for v in views], dim=0)

        if self.training:
            projections = [self.projector(emb) for emb in embeddings]
            out["loss"] = self.simclr_loss(projections[0], projections[1])
            self.log(
                f"{stage}/loss",
                out["loss"],
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
    else:
        out["embedding"] = _extract_embedding(
            self.backbone.forward_features(batch["image"]), pool_strategy
        )
        if "label" in batch:
            out["label"] = batch["label"]

    return out
