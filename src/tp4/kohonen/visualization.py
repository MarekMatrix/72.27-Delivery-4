from src.tp4.kohonen.topology import hex_coordinates
from matplotlib.patches import RegularPolygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_som_grid(values, grid_type="rectangular",
                  title="", colorbar_label="Value"):
    width, height = values.shape[:2]

    fig, ax = plt.subplots(figsize=(9, 7))

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

    plt.tight_layout()
    plt.show()


def visualize_som_weights(weights, feature, grid_type="rectangular"):
    plot_som_grid(
        weights[:, :, feature],
        grid_type=grid_type,
        title=f"SOM weights - Feature {feature}",
        colorbar_label="Standardized feature value"
    )

def visualize_u_matrix(u_matrix, grid_type="rectangular"):
    plot_som_grid(
        u_matrix,
        grid_type=grid_type,
        title=f"U-Matrix ({grid_type})",
        colorbar_label="Mean neighbor distance"
    )