import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def _cluster_color_map(labels, palette_name="tab20"):
    """Deterministic mapping from discrete cluster id to RGB color."""
    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    palette = sns.color_palette(palette_name, n_colors=max(1, len(unique_labels)))
    # Keys match what _format_cluster_label produces (no sample count here)
    return {
        _format_cluster_label(label): (
            palette[i],
            np.sum(labels == label),   # store count separately
        )
        for i, label in enumerate(unique_labels)
    }



def get_category_palette(category_name):
    """Deterministically assign a color palette name based on the string category_name."""
    if not category_name:
        return "tab10"
        
    category_name = str(category_name).lower()
    if "syllable" in category_name:
        return "tab20"  # Needs many colors usually
        
    # Pick from a list of nice categorical palettes
    palettes = [
        "Set1", "Set2", "Set3", "Dark2", "tab10",
        "Accent", "Pastel1", "Pastel2"
    ]
    import hashlib
    # Deterministic hash to consistently pick the same index
    idx = int(hashlib.md5(category_name.encode('utf-8')).hexdigest(), 16) % len(palettes)
    return palettes[idx]

def _format_cluster_label(value):
    """Format cluster label into canonical string key used by color maps."""
    return f"cluster {value}"


def plot_2D(
    x,
    label,
    dims=(0, 1),
    is_discrete=True,
    title=None,
    figsize=(8, 6),
    output_path=None,
    max_points=100000,
    random_state=42,
    color_map=None,
):
    x = np.asarray(x)
    label = np.asarray(label)

    # Build color map before sampling so colors are consistent across plots.
    # color_map: { "cluster N": (rgb, n_samples) }
    if is_discrete and color_map is None:
        color_map = _cluster_color_map(label, palette_name="tab10")

    if len(x) > max_points:
        np.random.seed(random_state)
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        label = label[indices]

    plt.figure(figsize=figsize)

    if dims[0] >= x.shape[1] or dims[1] >= x.shape[1]:
        print(f"Error: dims {dims} out of range for x with shape {x.shape} - defaulting to (0,1)")
        dims = (0, 1)

    if is_discrete:
        # Palette and hue_order both use the plain "cluster N" keys
        palette = {k: v[0] for k, v in color_map.items()}
        hue_order = sorted(palette.keys())
        hue_values = np.array([_format_cluster_label(v) for v in label])

        sns.scatterplot(
            x=x[:, dims[0]],
            y=x[:, dims[1]],
            hue=hue_values,
            hue_order=hue_order,
            palette=palette,
            s=10,
            alpha=0.7,
        )

        # Rebuild legend with sample counts in the label text
        handles = [
            mlines.Line2D(
                [], [],
                marker="o", linestyle="",
                color=color_map[k][0],
                markersize=10,
                label=f"{k} (n={color_map[k][1]})",
            )
            for k in hue_order
        ]
        legend = plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
        for handle in legend.legend_handles:
            handle.set_markersize(10)
    else:
        plt.scatter(x[:, dims[0]], x[:, dims[1]], c=label, cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(label=label.name if hasattr(label, "name") else "value")

    plt.xlabel(f"Dim {dims[0]}")
    plt.ylabel(f"Dim {dims[1]}")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        print(f"Saved 2D plot to {output_path}")
    plt.close()