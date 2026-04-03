"""
sae_decode.py - Pass activations through an SAE encoder to get sparse features,
then decode back to measure reconstruction quality.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sae_lens import SAE


@dataclass
class SAEResult:
    """Result of encoding activations through an SAE."""

    feature_acts: torch.Tensor
    reconstruction: torch.Tensor
    mse: float
    l0: float
    sparsity: float


@dataclass
class SparseFeatureActs:
    """Compact sparse representation of SAE feature activations for one prompt."""

    token_indices: np.ndarray
    feature_indices: np.ndarray
    values: np.ndarray
    seq_len: int
    d_sae: int

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    def save(self, path: str | Path) -> None:
        """Persist sparse activations as a compressed NumPy archive."""
        np.savez_compressed(
            path,
            token_indices=self.token_indices,
            feature_indices=self.feature_indices,
            values=self.values,
            seq_len=np.array([self.seq_len], dtype=np.int32),
            d_sae=np.array([self.d_sae], dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SparseFeatureActs":
        """Load a sparse feature archive written by save()."""
        with np.load(path) as data:
            return cls(
                token_indices=data["token_indices"],
                feature_indices=data["feature_indices"],
                values=data["values"],
                seq_len=int(data["seq_len"][0]),
                d_sae=int(data["d_sae"][0]),
            )

    def to_dense(self, dtype: np.dtype = np.float32) -> np.ndarray:
        """Reconstruct the dense [seq_len, d_sae] matrix when needed downstream."""
        dense = np.zeros((self.seq_len, self.d_sae), dtype=dtype)
        if self.nnz:
            dense[self.token_indices, self.feature_indices] = self.values.astype(dtype, copy=False)
        return dense


def sparsify_feature_acts(feature_acts: torch.Tensor) -> SparseFeatureActs:
    """Convert dense SAE activations into a compact sparse format.

    Args:
        feature_acts: [batch, seq_len, d_sae] or [seq_len, d_sae] tensor.

    Returns:
        SparseFeatureActs containing only non-zero feature entries.
    """
    if feature_acts.ndim == 3:
        if feature_acts.shape[0] != 1:
            raise ValueError("Expected batch size 1 when sparsifying batched feature activations")
        acts = feature_acts[0]
    elif feature_acts.ndim == 2:
        acts = feature_acts
    else:
        raise ValueError(f"Expected 2D or 3D feature activations, got shape {tuple(feature_acts.shape)}")

    acts_cpu = acts.detach().cpu()
    seq_len, d_sae = acts_cpu.shape
    token_indices, feature_indices = (acts_cpu > 0).nonzero(as_tuple=True)
    values = acts_cpu[token_indices, feature_indices]

    return SparseFeatureActs(
        token_indices=token_indices.numpy().astype(np.uint16, copy=False),
        feature_indices=feature_indices.numpy().astype(np.uint16, copy=False),
        values=values.numpy().astype(np.float16, copy=False),
        seq_len=int(seq_len),
        d_sae=int(d_sae),
    )


class GemmaScopeSAE:
    """Wrapper around a Gemma Scope SAE for encoding and decoding activations."""

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
        """Encode activations into sparse feature space."""
        x = activations.to(self.device).to(self.dtype)
        with torch.no_grad():
            return self.sae.encode(x)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        with torch.no_grad():
            return self.sae.decode(feature_acts)

    def forward(self, activations: torch.Tensor) -> SAEResult:
        """Full encode -> decode forward pass with metrics."""
        x = activations.to(self.device).to(self.dtype)

        with torch.no_grad():
            feature_acts = self.sae.encode(x)
            reconstruction = self.sae.decode(feature_acts)

        mse = ((x - reconstruction) ** 2).mean().item()
        active_per_pos = (feature_acts > 0).float().sum(dim=-1)
        l0 = active_per_pos.mean().item()
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
        """Get the top-k most active features for a specific token position."""
        acts_at_pos = feature_acts[:, token_idx, :]
        return acts_at_pos.topk(k, dim=-1)

    def encode_sparse(self, activations: torch.Tensor) -> SparseFeatureActs:
        """Encode activations and return a compact sparse representation."""
        return sparsify_feature_acts(self.encode(activations))

    def to(self, device: str) -> "GemmaScopeSAE":
        """Move SAE to a different device."""
        self.sae = self.sae.to(device)
        self.device = device
        return self
