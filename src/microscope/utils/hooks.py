"""TransformerLens hook helpers.

All hook-point name strings are constructed from the templates defined here; downstream
modules never write a hook name as an inline string literal.
"""

from __future__ import annotations

from typing import Callable

# Hook-point templates. ``{layer}`` is filled in by the helpers below.
RESID_POST_HOOK_TEMPLATE: str = "blocks.{layer}.hook_resid_post"
RESID_PRE_HOOK_TEMPLATE: str = "blocks.{layer}.hook_resid_pre"
ATTN_Z_HOOK_TEMPLATE: str = "blocks.{layer}.attn.hook_z"


def resid_post_name(layer: int) -> str:
    """Return the TransformerLens hook name for the post-block residual stream at ``layer``."""
    return RESID_POST_HOOK_TEMPLATE.format(layer=layer)


def resid_pre_name(layer: int) -> str:
    """Return the TransformerLens hook name for the pre-block residual stream at ``layer``."""
    return RESID_PRE_HOOK_TEMPLATE.format(layer=layer)


def attn_z_name(layer: int) -> str:
    """Return the hook name for per-head attention output (``hook_z``) at ``layer``."""
    return ATTN_Z_HOOK_TEMPLATE.format(layer=layer)


def get_activation(model, tokens, hook_name: str):
    """Run ``model`` on ``tokens`` with caching and return the activation at ``hook_name``.

    Parameters
    ----------
    model : HookedTransformer
    tokens : torch.Tensor of shape (batch, seq)
    hook_name : str
        A hook-point name (e.g. from :func:`resid_post_name`).
    """
    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
    return cache[hook_name]


def make_replace_hook(replacement, position: int | None = None) -> Callable:
    """Build a TransformerLens forward hook that overwrites an activation in place.

    Parameters
    ----------
    replacement : torch.Tensor
        Tensor of the same shape as the activation being hooked. When ``position`` is given
        only the slice at that sequence index is copied across (used by activation patching).
    position : int or None
        If ``None`` the whole activation is replaced; otherwise only ``[:, position, :]``.
    """

    def hook(activation, hook):  # noqa: ANN001 - signature fixed by TransformerLens
        if position is None:
            activation[...] = replacement
        else:
            activation[:, position, :] = replacement[:, position, :]
        return activation

    return hook
