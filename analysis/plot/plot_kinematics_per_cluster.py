import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_kinematics_per_cluster(kinematics_results, output_path):
    if not kinematics_results:
        return

    clusters = sorted(kinematics_results.keys())
    sample_cluster = clusters[0]
    features = [f for f in kinematics_results[sample_cluster].keys()
                if kinematics_results[sample_cluster][f]["mean"] is not None]

    if not features:
        return

    n_features = len(features)
    n_clusters = len(clusters)

    fig, axes = plt.subplots(
        n_features, 1,
        figsize=(max(6, n_clusters * 1.2), 3.5 * n_features),
        squeeze=False
    )

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for f_idx, feature in enumerate(features):
        ax = axes[f_idx][0]

        means = []
        stds = []
        valid_clusters = []

        for cluster in clusters:
            cluster_data = kinematics_results[cluster].get(feature, {})
            mean = cluster_data.get("mean")
            std = cluster_data.get("std")
            if mean is not None:
                means.append(mean)
                stds.append(std if std is not None else 0.0)
                valid_clusters.append(cluster)

        x = np.arange(len(valid_clusters))

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors[:len(valid_clusters)],
            width=0.6,
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "black", "alpha": 0.7},
            edgecolor="white",
            linewidth=0.5
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Cluster {c}" for c in valid_clusters], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(feature.replace("_", " ").capitalize(), fontsize=10)
        ax.set_title(feature.replace("_", " ").capitalize(), fontsize=11, fontweight="normal", pad=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        for bar, mean_val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05 if stds else bar.get_height() * 1.02,
                f"{mean_val:.2f}",
                ha="center", va="bottom", fontsize=8,
                color="dimgray"
            )

    fig.suptitle("Mean kinematics per cluster (± std)", fontsize=13, fontweight="normal", y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved kinematics plot to {output_path}")