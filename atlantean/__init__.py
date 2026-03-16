"""Minimal package wrapper for the Atlantean field engine.

This package is intentionally lightweight and primarily exists so that
scripts can import the engine modules via a consistent package path.
"""

from .atlantean_engine import (
    AtlanteanFieldNetAdvanced,
    train_with_l1,
    dodecagonal_mask,
    spectral_target_loss,
    lightweight_infer,
    encode_prompt_into_phi,
)

__all__ = [
    "AtlanteanFieldNetAdvanced",
    "train_with_l1",
    "dodecagonal_mask",
    "spectral_target_loss",
    "lightweight_infer",
    "encode_prompt_into_phi",
]
