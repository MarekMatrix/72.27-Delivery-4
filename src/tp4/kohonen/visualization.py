from src.tp4.kohonen.topology import hex_coordinates
from matplotlib.patches import RegularPolygon, Rectangle
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import seaborn as sns


def plot_som_grid(values, grid_type="rectangular",
                  title="", colorbar_label="Value",
                  labels=None, ax=None, figsize=(9, 7), label_fontsize=6):
    """Draw one value per neuron. `labels` maps (i, j) -> text drawn on that cell.
    Pass `ax` to draw into an existing axes (small multiples); the caller then
    owns layout and plt.show()."""
    width, height = values.shape[:2]

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for i in range(width):
        for j in range(height):
            color = cmap(norm(values[i, j]))

            if grid_type == "hexagonal":
                pos = hex_coordinates(i, j)
                patch = RegularPolygon(
                    pos,
                    numVertices=6,
                    radius=1 / np.sqrt(3),
                    orientation=0,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5
                )
                ax.add_patch(patch)

            elif grid_type == "rectangular":
                patch = Rectangle(
                    (i - 0.5, j - 0.5),
                    1, 1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5
                )
                ax.add_patch(patch)

            else:
                raise ValueError("Unknown grid type")

    if labels:
        for (i, j), text in labels.items():
            x, y = hex_coordinates(i, j) if grid_type == "hexagonal" else (i, j)
            ax.text(x, y, text, ha="center", va="center", fontsize=label_fontsize,
                    color="white",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    if grid_type == "hexagonal":
        coords = np.array([
            hex_coordinates(i, j)
            for i in range(width)
            for j in range(height)
        ])
        ax.set_xlim(coords[:, 0].min() - 0.6,
                    coords[:, 0].max() + 0.6)
        ax.set_ylim(coords[:, 1].min() - 0.6,
                    coords[:, 1].max() + 0.6)
    else:
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(title)

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    fig.colorbar(scalar_map, ax=ax, label=colorbar_label,
                 shrink=0.8)

    if own_figure:
        plt.tight_layout()
        plt.show()


def visualize_som_weights(weights, feature, grid_type="rectangular",
                          feature_name=None, ax=None):
    plot_som_grid(
        weights[:, :, feature],
        grid_type=grid_type,
        title=feature_name or f"SOM weights - Feature {feature}",
        colorbar_label="Standardized feature value",
        ax=ax,
    )

def visualize_u_matrix(u_matrix, grid_type="rectangular"):
    plot_som_grid(
        u_matrix,
        grid_type=grid_type,
        title=f"U-Matrix ({grid_type})",
        colorbar_label="Mean neighbor distance"
    )