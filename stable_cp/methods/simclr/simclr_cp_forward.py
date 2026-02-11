import torch


def _extract_embedding(embedding):
    # Extract embedding from backbone output (handles ViT and CNN)
    if embedding.dim() == 3:
        return embedding[:, 0, :]  # TIMM ViT CLS token
    return embedding  # Already 2D (ResNet or pooled)


def _get_views_list(batch):
    # Convert multi-view batch to list of views
    if isinstance(batch, list):
        return batch
    elif isinstance(batch, dict) and "image" not in batch:
        views = [v for v in batch.values() if isinstance(v, dict) and "image" in v]
        return views if views else None
    return None


def simclr_cp_forward(self, batch, stage):
    # SimCLR forward with ViT backbone support (CLS token extraction)
    out = {}
    views = _get_views_list(batch)

    if views is not None:
        if len(views) != 2:
            raise ValueError(f"SimCLR requires 2 views, got {len(views)}")

        embeddings = [_extract_embedding(self.backbone(v["image"])) for v in views]
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
        out["embedding"] = _extract_embedding(self.backbone(batch["image"]))
        if "label" in batch:
            out["label"] = batch["label"]

    return out
