"""Utility helpers (TransformerLens hook points and forward-hook factories)."""

from __future__ import annotations

from .hooks import (
    ATTN_Z_HOOK_TEMPLATE,
    RESID_POST_HOOK_TEMPLATE,
    RESID_PRE_HOOK_TEMPLATE,
    attn_z_name,
    get_activation,
    make_replace_hook,
    resid_post_name,
    resid_pre_name,
)

__all__ = [
    "RESID_POST_HOOK_TEMPLATE",
    "RESID_PRE_HOOK_TEMPLATE",
    "ATTN_Z_HOOK_TEMPLATE",
    "resid_post_name",
    "resid_pre_name",
    "attn_z_name",
    "get_activation",
    "make_replace_hook",
]
