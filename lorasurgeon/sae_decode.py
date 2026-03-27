"""
sae_decode.py — Pass activations through SAE encoder to get sparse features,
then decode back to measure reconstruction quality.
"""

import torch
from sae_lens import SAE
from dataclasses import dataclass


@dataclass
class SAEResult:
    """Result of encoding activations through an SAE."""
    feature_acts: torch.Tensor       # [batch, seq_len, d_sae] — sparse feature activations
    reconstruction: torch.Tensor     # [batch, seq_len, d_in]  — decoded reconstruction
    mse: float                       # mean squared error between input and reconstruction
    l0: float                        # average number of active features per position
    sparsity: float                  # fraction of features active (across all positions)


class GemmaScopeSAE:
    """Wrapper around a Gemma Scope SAE for encoding and decoding activations.

    Loads from SAELens and provides encode/decode/full-forward methods
    with reconstruction loss computation.
    """

    DEFAULT_RELEASE = "gemma-scope-2b-pt-res"
    DEFAULT_SAE_ID = "layer_12/width_16k/average_l0_82"

    def __init__(
        self,
        release: str = DEFAULT_RELEASE,
        sae_id: str = DEFAULT_SAE_ID,
        device: str = "cuda",
    ):
        self.release = release
        self.sae_id = sae_id
        self.device = device
        self.sae = self._load(release, sae_id, device)

    @staticmethod
    def _load(release: str, sae_id: str, device: str) -> SAE:
        sae, _cfg, _sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )
        return sae

    @property
    def d_in(self) -> int:
        return self.sae.cfg.d_in

    @property
    def d_sae(self) -> int:
        return self.sae.cfg.d_sae

    @property
    def dtype(self):
        return self.sae.dtype

    @property
    def layer(self) -> int:
        """Extract layer number from sae_id (e.g. 'layer_12/...' -> 12)."""
        return int(self.sae_id.split("/")[0].split("_")[1])

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """Encode activations into sparse feature space.

        Args:
            activations: [batch, seq_len, d_in] residual stream activations.

        Returns:
            Feature activations [batch, seq_len, d_sae], mostly zeros.
        """
        x = activations.to(self.device).to(self.dtype)
        with torch.no_grad():
            return self.sae.encode(x)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space.

        Args:
            feature_acts: [batch, seq_len, d_sae] sparse feature activations.

        Returns:
            Reconstructed activations [batch, seq_len, d_in].
        """
        with torch.no_grad():
            return self.sae.decode(feature_acts)

    def forward(self, activations: torch.Tensor) -> SAEResult:
        """Full encode → decode forward pass with metrics.

        Args:
            activations: [batch, seq_len, d_in] residual stream activations.

        Returns:
            SAEResult with features, reconstruction, and quality metrics.
        """
        x = activations.to(self.device).to(self.dtype)

        with torch.no_grad():
            feature_acts = self.sae.encode(x)
            reconstruction = self.sae.decode(feature_acts)

        # Reconstruction MSE
        mse = ((x - reconstruction) ** 2).mean().item()

        # L0: average number of active (nonzero) features per position
        active_per_pos = (feature_acts > 0).float().sum(dim=-1)  # [batch, seq_len]
        l0 = active_per_pos.mean().item()

        # Sparsity: fraction of all feature entries that are active
        sparsity = (feature_acts > 0).float().mean().item()

        return SAEResult(
            feature_acts=feature_acts.cpu(),
            reconstruction=reconstruction.cpu(),
            mse=mse,
            l0=l0,
            sparsity=sparsity,
        )

    def top_features(
        self,
        feature_acts: torch.Tensor,
        k: int = 10,
        token_idx: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the top-k most active features for a specific token position.

        Args:
            feature_acts: [batch, seq_len, d_sae] from encode().
            k: Number of top features to return.
            token_idx: Which token position (-1 = last).

        Returns:
            (values, indices) each of shape [batch, k].
        """
        acts_at_pos = feature_acts[:, token_idx, :]  # [batch, d_sae]
        return acts_at_pos.topk(k, dim=-1)

    def to(self, device: str) -> "GemmaScopeSAE":
        """Move SAE to a different device."""
        self.sae = self.sae.to(device)
        self.device = device
        return self
