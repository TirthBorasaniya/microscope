"""Feature activation visualization (heatmaps).

Plotting dependencies (matplotlib / seaborn) are part of the optional ``viz`` extra and are
imported lazily so the rest of the package does not require them.
"""

from __future__ import annotations

from typing import Any


def plot_feature_activation_heatmap(
    activation_matrix,
    token_labels: list[str] | None = None,
    feature_labels: list[Any] | None = None,
    output_path: str | None = None,
):
    """Plot a (tokens x features) activation matrix as a heatmap.

    Parameters
    ----------
    activation_matrix : 2D array-like of shape (n_tokens, n_features)
    token_labels : optional row (token) labels
    feature_labels : optional column (feature) labels
    output_path : if given, the figure is saved there; otherwise it is returned for display.

    Returns
    -------
    The matplotlib ``Axes`` for the heatmap.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    matrix = np.asarray(activation_matrix, dtype=float)
    fig, ax = plt.subplots()
    sns.heatmap(
        matrix,
        xticklabels=feature_labels if feature_labels is not None else "auto",
        yticklabels=token_labels if token_labels is not None else "auto",
        cmap="viridis",
        ax=ax,
    )
    ax.set_xlabel("feature")
    ax.set_ylabel("token")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    return ax
