import lightning as pl
import torch


class LeJEPAMetricsCallback(pl.Callback):
    # Log LeJEPA/SIGReg diagnostic metrics
    def __init__(
        self,
        log_every_n_steps: int = 50,
        compute_covariance: bool = False,
        prefix: str = "lejepa",
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.compute_covariance = compute_covariance
        self.prefix = prefix

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log embedding statistics after each training batch
        if batch_idx % self.log_every_n_steps != 0:
            return

        # Get embeddings from outputs
        if not isinstance(outputs, dict) or "embedding" not in outputs:
            return

        embedding = outputs["embedding"].detach()

        with torch.no_grad():
            self._log_embedding_stats(pl_module, embedding, "train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Log embedding statistics during validation
        if batch_idx != 0:  # Only log first val batch to reduce overhead
            return

        if not isinstance(outputs, dict) or "embedding" not in outputs:
            return

        embedding = outputs["embedding"].detach()

        with torch.no_grad():
            self._log_embedding_stats(pl_module, embedding, "val")

    def _log_embedding_stats(self, pl_module, embedding: torch.Tensor, stage: str):
        # Compute and log embedding statistics
        # embedding shape: [batch_size, embedding_dim]
        batch_size, emb_dim = embedding.shape

        # Per-dimension statistics
        dim_mean = embedding.mean(dim=0)  # [emb_dim]
        dim_var = embedding.var(dim=0)  # [emb_dim]

        # Aggregate statistics
        mean_of_means = dim_mean.mean()  # Should → 0
        mean_of_vars = dim_var.mean()  # Should → 1
        std_of_means = dim_mean.std()  # Spread of dimension means
        std_of_vars = dim_var.std()  # Spread of dimension variances

        # Embedding norms
        embedding_norms = torch.norm(embedding, dim=1)  # [batch_size]
        mean_norm = embedding_norms.mean()
        std_norm = embedding_norms.std()

        # Log metrics
        pl_module.log(
            f"{stage}/{self.prefix}/mean_of_dim_means", mean_of_means, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/mean_of_dim_vars", mean_of_vars, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/std_of_dim_means", std_of_means, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/std_of_dim_vars", std_of_vars, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/embedding_norm_mean", mean_norm, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/embedding_norm_std", std_norm, sync_dist=True
        )

        # Deviation from N(0,I) target
        mean_deviation = mean_of_means.abs()  # Distance from 0
        var_deviation = (mean_of_vars - 1).abs()  # Distance from 1
        pl_module.log(
            f"{stage}/{self.prefix}/mean_deviation", mean_deviation, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/var_deviation", var_deviation, sync_dist=True
        )

        # Covariance statistics (optional, more expensive)
        if self.compute_covariance and batch_size > 1:
            self._log_covariance_stats(pl_module, embedding, stage)

    def _log_covariance_stats(self, pl_module, embedding: torch.Tensor, stage: str):
        # Compute covariance matrix statistics
        # Center the embeddings
        centered = embedding - embedding.mean(dim=0, keepdim=True)

        # Compute covariance matrix: (1/n) * X^T @ X
        batch_size = embedding.shape[0]
        cov = (centered.T @ centered) / (batch_size - 1)

        # Diagonal statistics (should be ~1)
        diag = torch.diag(cov)
        diag_mean = diag.mean()
        diag_std = diag.std()

        # Off-diagonal statistics (should be ~0)
        mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
        off_diag = cov[mask]
        off_diag_mean = off_diag.abs().mean()
        off_diag_max = off_diag.abs().max()

        pl_module.log(f"{stage}/{self.prefix}/cov_diag_mean", diag_mean, sync_dist=True)
        pl_module.log(f"{stage}/{self.prefix}/cov_diag_std", diag_std, sync_dist=True)
        pl_module.log(
            f"{stage}/{self.prefix}/cov_offdiag_abs_mean", off_diag_mean, sync_dist=True
        )
        pl_module.log(
            f"{stage}/{self.prefix}/cov_offdiag_abs_max", off_diag_max, sync_dist=True
        )
