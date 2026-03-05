import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import umap
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
import seaborn as sns

from tqdm import tqdm

## CLUSTER PLOTS

def scatter_layer_embeddings(
    list_of_embeddings: list,
    list_of_colors: list = None,
    keypoints=None,
    figsize=(12, 4),
    title=None,
    d3=False,
    interactive=False,
    colortime=False,
    center_idx=9,
    colorspeed=False
):
    """
    Plots embeddings for each layer using optional per-point colors.

    Args:
        list_of_embeddings: list of np.ndarray, each shape (N, D)
        list_of_colors: list of np.ndarray, each shape (N, 3 or 4), optional
        figsize: figure size
        title: optional figure title
        d3: plot 3D if True
        interactive: use plotly for 3D visualization if True
        colortime: if True, color points by frame index (time). Overrides list_of_colors.
        colorspeed: if True, color points by speed. Overrides list_of_colors.
    """
    num_layers = len(list_of_embeddings)

    if list_of_colors is not None and len(list_of_colors) != num_layers:
        raise ValueError("list_of_colors must match list_of_embeddings length")

    fig = plt.figure(figsize=figsize)

    for i, embeddings in enumerate(list_of_embeddings):
        # Determine colors to use
        if colortime:
            # Create time-based colors (frame index)
            frame_colors = plt.cm.viridis(np.linspace(0, 1, len(embeddings)))

        elif colorspeed:
            speed = compute_speed(keypoints, center_idx)
            # Choose a colormap you like, e.g., 'plasma', 'coolwarm', 'cividis', etc.
            cmap = plt.cm.seismic
            
            # normalize speed to [0,1] for colormap
            speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed) + 1e-8)

            # Map speeds to RGBA colors
            frame_colors = cmap(speed)
        elif list_of_colors is not None:
            frame_colors = list_of_colors[i]
        else:
            frame_colors = None

        if d3:
            if interactive:
                fig = go.Figure(
                    go.Scatter3d(
                        x=embeddings[:,0],
                        y=embeddings[:,1],
                        z=embeddings[:,2],
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=frame_colors 
                        )
                    )
                )
                fig.update_layout(title=f"Layer {i} Embeddings", )
                fig.show()
                if i ==len(list_of_embeddings) - 1:
                    return
                continue

            else:
                ax = fig.add_subplot(1, num_layers, i + 1, projection="3d")
                if frame_colors is not None:
                    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=frame_colors, alpha=0.5, s=1)
                else:
                    ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], alpha=0.5, s=1)
                ax.set_zlabel("Dim 2")
        else:
            ax = fig.add_subplot(1, num_layers, i + 1)
            if frame_colors is not None:
                if colortime:
                    # Use frame index as scalar for coloring
                    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=np.arange(len(embeddings)), cmap="viridis", alpha=0.5, s=1)
                    if i == num_layers - 1:
                        plt.colorbar(sc, ax=ax, label="Frame Index")
                elif colorspeed:
                    sc = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=speed, cmap="seismic", alpha=0.5, s=1)
                    if i == num_layers - 1:
                        plt.colorbar(sc, ax=ax, label="Speed")
                else:
                    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=frame_colors, alpha=0.5, s=1)
            else:
                ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.5, s=1)

        ax.set_xlabel("Dim 0")
        ax.set_ylabel("Dim 1")
        ax.set_title(f"Layer {i} Embeddings")

    if title:
        plt.suptitle(title)

    if list_of_colors is not None and not colortime:
        for i, (ax, layer_colors) in enumerate(zip(fig.axes, list_of_colors)):
            unique_colors = np.unique(layer_colors, axis=0)

            handles = [
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=col,
                    markeredgecolor="k",
                    markersize=6,
                )
                for col in unique_colors
            ]

            labels = [f"Cluster {j}" for j in range(len(unique_colors))]

            ax.legend(
                handles,
                labels,
                title=f"Layer {i} Clusters",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                frameon=False,
                ncol=min(4, len(unique_colors)),
                fontsize=8,
                title_fontsize=9,
            )
        
    plt.tight_layout()
    plt.show()


def temporal_cluster_plot(list_of_cluster_colors: list, figsize=(18, 3), layer_labels=None):
    """
    Plots temporal cluster colors for each layer as a heatmap.

    Args:
        list_of_cluster_colors: list of np.ndarray, each shape (num_frames, 3 or 4), colors per frame
        figsize: figure size
        layer_labels: optional list of layer names
    """
    num_layers = len(list_of_cluster_colors)
    # Stack into shape (num_layers, num_frames, 3/4)
    labels_img = np.stack(list_of_cluster_colors, axis=0)

    plt.figure(figsize=figsize)
    plt.imshow(labels_img, aspect="auto", interpolation="nearest")

    # Y-axis ticks
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(num_layers)]
    plt.yticks(np.arange(num_layers), layer_labels)

    plt.xlabel("Frame index")
    plt.title("Temporal Cluster Attribution Across Layers")
    plt.tight_layout()
    plt.show()

def cluster_distribution_plot(parent_color_labels, child_color_labels, parent_layer_name="Parent Layer", child_layer_name="Child Layer", figsize=(5, 2)):
    """
    Plots the distribution of child cluster colors within each parent cluster as a stacked bar chart.

    Args:
        parent_color_labels: np.ndarray of shape (num_frames, 3 or 4), colors for parent clusters
        child_color_labels: np.ndarray of shape (num_frames, 3 or 4), colors for child clusters
        parent_layer_name: name for parent layer (for legend)
        child_layer_name: name for child layer (for legend)
    """
    # Get unique parent clusters
    unique_parent_colors = np.unique(parent_color_labels, axis=0)
    unique_child_colors = np.unique(child_color_labels, axis=0)

    # Count occurrences of each child color within each parent color
    distribution = []
    for parent_col in unique_parent_colors:
        mask = np.all(parent_color_labels == parent_col, axis=1)
        child_counts = []
        for child_col in unique_child_colors:
            count = np.sum(np.all(child_color_labels[mask] == child_col, axis=1))
            child_counts.append(count)
        distribution.append(child_counts)

    distribution = np.array(distribution)  # shape (num_parent_clusters, num_child_clusters)

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(distribution.shape[0])
    for i in range(distribution.shape[1]):
        ax.bar(range(distribution.shape[0]), distribution[:, i], bottom=bottom, color=unique_child_colors[i], label=f"{child_layer_name} Cluster {i}")
        bottom += distribution[:, i]

    ax.set_xticks(range(distribution.shape[0]))
    ax.set_xlabel(f"{parent_layer_name} Clusters ID")
    ax.set_ylabel("Frame Count")
    ax.set_title("Distribution of Child Clusters Within Parent Clusters")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_mean_duration_per_cluster_colors(list_of_frame_colors: list, layer_labels=None, figsize=(8,4), per_cluster=True):
    """
    Plots mean duration of consecutive frames with the cluster colors provided directly.

    Args:
        list_of_frame_colors: list of arrays, each shape (num_frames, 3 or 4), per-frame colors per layer
        layer_labels: optional list of layer names
        figsize: figure size
    """
    num_layers = len(list_of_frame_colors)
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(num_layers)]
    
    all_durations = []
    all_colors = []

    # compute consecutive durations of the same color
    for frame_colors in list_of_frame_colors:
        frame_colors = np.array(frame_colors)
        durations = []
        colors = []

        # find runs of consecutive identical colors
        i = 0
        while i < len(frame_colors):
            color = frame_colors[i]
            j = i + 1
            while j < len(frame_colors) and np.all(frame_colors[j] == color):
                j += 1
            durations.append(j - i)
            colors.append(color)
            i = j

        if per_cluster:
            # average durations per unique color
            unique_colors = np.unique(colors, axis=0)
            avg_durations = []
            for uc in unique_colors:
                mask = np.all(np.array(colors) == uc, axis=1)
                avg_durations.append(np.mean(np.array(durations)[mask]))
            durations = avg_durations
            colors = unique_colors
        all_durations.append(durations)
        all_colors.append(colors)

    plt.figure(figsize=figsize)
    plt.boxplot(all_durations, labels=layer_labels)
    plt.ylabel("Mean Duration (frames)")
    plt.title(f"Mean Duration of Consecutive Frames {'per Cluster' if per_cluster else 'Overall'}")

    # add dots colored according to cluster colors
    for i, (durations, colors) in enumerate(zip(all_durations, all_colors)):
        x = np.random.normal(i+1, 0.04, size=len(durations))  # jitter
        for j, dur in enumerate(durations):
            plt.scatter(x[j], dur, color=colors[j], alpha=0.7, edgecolors='k', s=50)

    plt.tight_layout()
    plt.show()


def compute_mean_cosine_distance_between_color_clusters(embeddings, frame_colors):
    """
    Compute mean cosine distances between unique colors (clusters) given per-frame colors.

    Args:
        embeddings: np.ndarray, shape (num_frames, embedding_dim)
        frame_colors: np.ndarray, shape (num_frames, 3 or 4), color per frame (RGB or RGBA)
    
    Returns:
        mean_distances: np.ndarray, shape (num_unique_colors, num_unique_colors)
        unique_colors: np.ndarray, list of unique colors
    """
    frame_colors = np.array(frame_colors)
    unique_colors, inverse_indices = np.unique(frame_colors, axis=0, return_inverse=True)
    n_colors = len(unique_colors)
    mean_distances = np.zeros((n_colors, n_colors))

    for i in range(n_colors):
        indices_i = np.where(inverse_indices == i)[0]
        for j in range(i, n_colors):
            indices_j = np.where(inverse_indices == j)[0]
            if len(indices_i) == 0 or len(indices_j) == 0:
                mean_distances[i, j] = np.nan
                mean_distances[j, i] = np.nan
                continue
            dist = cosine_distances(embeddings[indices_i], embeddings[indices_j])
            mean_distances[i, j] = np.mean(dist)
            mean_distances[j, i] = mean_distances[i, j]  # symmetric
    return mean_distances, unique_colors

def plot_mean_cosine_distance_layers(
    list_of_embeddings, 
    list_of_frame_colors, 
    layer_labels=None, 
    figsize=(12, 4), 
    cmap="Blues"
):
    """
    Plot heatmaps of mean cosine distances between clusters (colors) for multiple layers,
    coloring axis ticks to match the cluster colors.

    Args:
        list_of_embeddings: list of np.ndarray, embeddings per layer
        list_of_frame_colors: list of np.ndarray, per-frame colors per layer
        layer_labels: optional list of layer names
        figsize: figure size
        cmap: colormap
    """
    num_layers = len(list_of_embeddings)
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(num_layers)]

    # First, compute all mean distance matrices
    mean_distances_list = []
    for emb, colors in zip(list_of_embeddings, list_of_frame_colors):
        mean_distances, unique_colors = compute_mean_cosine_distance_between_color_clusters(emb, colors)
        mean_distances_list.append(mean_distances)

    # Determine global min/max across all layers
    all_values = np.concatenate([md.flatten() for md in mean_distances_list])
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)

    plt.figure(figsize=figsize)
    for i, mean_distances in enumerate(mean_distances_list):
        ax = plt.subplot(1, num_layers, i + 1)
        im = ax.imshow(mean_distances, cmap=cmap, vmin=vmin, vmax=vmax)

        # Use numeric ticks but color them according to the cluster color
        unique_colors = np.unique(list_of_frame_colors[i], axis=0)
        ticks = np.arange(len(unique_colors))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{idx}" for idx in ticks])
        ax.set_yticklabels([f"{idx}" for idx in ticks])
        for tick_label, col in zip(ax.get_xticklabels(), unique_colors):
            tick_label.set_color(col)
        for tick_label, col in zip(ax.get_yticklabels(), unique_colors):
            tick_label.set_color(col)

        ax.set_title(layer_labels[i])

    # Add a single colorbar for all subplots
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax, label="Mean Cosine Distance")

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
    plt.suptitle("Mean Cosine Distance Between Clusters per Layer", y=1.02)
    plt.show()



def plot_transition_matrices(
    list_of_cluster_colors,  # list of arrays (num_frames, 3/4) per layer
    layer_labels=None,
    figsize=(12, 4),
    cmap='Blues',
    logscale=False
):
    """
    Plots cluster-to-cluster transition matrices from per-frame cluster colors.

    Args:
        list_of_cluster_colors: list of np.ndarray, each shape (num_frames, 3 or 4), cluster colors per frame
        layer_labels: optional list of layer names
        figsize: figure size
        cmap: matplotlib colormap
        logscale: if True, plot log(counts + 1e-8)
    """
    num_layers = len(list_of_cluster_colors)
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(num_layers)]

    transition_matrices = []
    min_val, max_val = np.inf, -np.inf

    for colors in list_of_cluster_colors:
        # Convert unique colors to integer cluster IDs
        unique_colors, cluster_ids = np.unique(colors, axis=0, return_inverse=True)
        n_clusters = len(unique_colors)

        # Build transition matrix
        T = np.zeros((n_clusters, n_clusters), dtype=float)
        for i in range(len(cluster_ids) - 1):
            T[cluster_ids[i], cluster_ids[i + 1]] += 1

        if logscale:
            T = np.log(T + 1e-8)

        transition_matrices.append((T, unique_colors))
        min_val = min(min_val, np.nanmin(T))
        max_val = max(max_val, np.nanmax(T))

    # Plot
    plt.figure(figsize=figsize)
    for i, (T, colors) in enumerate(transition_matrices):
        ax = plt.subplot(1, num_layers, i + 1)
        im = ax.imshow(T, cmap=cmap, vmin=min_val, vmax=max_val)
        n_clusters = T.shape[0]

        ax.set_xticks(np.arange(n_clusters))
        ax.set_yticks(np.arange(n_clusters))
        ax.set_xlabel("To Cluster")
        ax.set_ylabel("From Cluster")
        ax.set_title(layer_labels[i])

        # Color tick labels according to cluster colors
        for tick_label, col in zip(ax.get_xticklabels(), colors):
            tick_label.set_color(col)
        for tick_label, col in zip(ax.get_yticklabels(), colors):
            tick_label.set_color(col)

    # Single colorbar
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax, label="Log Count" if logscale else "Count")

    plt.tight_layout(rect=[0,0,0.9,1])
    plt.suptitle("Cluster-to-Cluster Transition Matrices", y=1.02)
    plt.show()




def plot_per_frame(
    keypoints=None, 
    list_of_frame_colors=None, 
    labels=None,
    keypoint_names=None,
    figsize=(18, 8)
):
    """
    Plot 

    Args
        keypoints: array of shape (num_frames, num_keypoints, 3)
        list_of_frame_colors: list of arrays of shape (num_frames, 3 or 4), per-frame colors for cluster labels
        keypoint_indices: which keypoints to plot (default: center, head, hands, feet)
        keypoint_names: names for legend
    """
    if keypoints is None and list_of_frame_colors is None and labels is None:
        raise ValueError("At least one of keypoints, list_of_frame_colors, or labels must be provided.")
    
    # find number of subplots needed
    n_subplots = sum([keypoints is not None, list_of_frame_colors is not None, labels is not None])
    fig, axs = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)

    if keypoints is not None:
        pca = PCA(n_components=1)
        n_frames, n_keypoints, _ = keypoints.shape
            
        for keypoint in np.arange(n_keypoints):
            data = keypoints[:,keypoint, :]
            component = pca.fit_transform(data)
            axs[0].plot(component + (keypoint+1), label=f"keypoint {keypoint}")
        axs[0].set_title("Keypoint 1st PCA per frame.")

    if list_of_frame_colors is not None:
        frame_colors = np.stack(list_of_frame_colors, axis=0)  # shape (num_layers, num_frames, 3/4)
        axs[1].imshow(frame_colors, aspect="auto", interpolation="nearest")
        # set y ticks to layer names if provided
        layer_labels = [f"Layer {i}" for i in range(frame_colors.shape[0])]
        axs[1].set_yticks(np.arange(frame_colors.shape[0]))
        axs[1].set_yticklabels(layer_labels)
        axs[1].set_xlabel("Frame Number")
        axs[1].set_title("Cluster Labels per Frame")

    if labels is not None:
        colors = plt.cm.cividis(np.linspace(0, 1, labels.shape[0]))
        for i, label in enumerate(labels):
            axs[2].scatter(np.arange(len(label)), label * (i+1), label=f"Label {i}", c=colors[i])
        axs[2].set_ylabel("Label (scaled for visibility)")
        axs[2].set_xlabel("Frame Number")
        axs[2].set_title("True Labels per Frame")

    # share x-axis
    axs[1].set_xlim(axs[0].get_xlim())
    axs[2].set_xlim(axs[0].get_xlim())

    plt.tight_layout()
    plt.show()


def plot_k_means_silhouettes(list_of_embeddings, k_range=(2,11), random_state=42):
    """ 
    Plots silhouette scores for KMeans clustering across a range of k values for multiple layers.
    """
    K_range = range(k_range[0], k_range[1])

    silhouette_scores_layer_0 = []
    silhouette_scores_layer_1 = []
    silhouette_scores_layer_2 = []

    for k in tqdm(K_range):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels_0 = kmeans.fit_predict(list_of_embeddings[0])
        labels_1 = kmeans.fit_predict(list_of_embeddings[1])
        labels_2 = kmeans.fit_predict(list_of_embeddings[2])
        silhouette_scores_layer_0.append(silhouette_score(list_of_embeddings[0], labels_0))
        silhouette_scores_layer_1.append(silhouette_score(list_of_embeddings[1], labels_1))
        silhouette_scores_layer_2.append(silhouette_score(list_of_embeddings[2], labels_2))

    # Choisissez le k avec le score le plus élevé
    sns.lineplot(x=K_range, y=silhouette_scores_layer_0, label='Layer 0', marker='o')
    sns.lineplot(x=K_range, y=silhouette_scores_layer_1, label='Layer 1', marker='o')
    sns.lineplot(x=K_range, y=silhouette_scores_layer_2, label='Layer 2', marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.xticks(K_range)
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Each Layer')
    plt.legend()
    plt.show()

def plot_dbscan_silhouettes(list_of_embeddings, eps_range=(0.5, 5.0), min_samples=5):
    """
    Plots silhouette scores for DBSCAN clustering across a range of eps values for multiple layers.
    """
    eps_values = np.arange(eps_range[0], eps_range[1], 1.0)

    silhouette_scores_layer_0 = []
    silhouette_scores_layer_1 = []
    silhouette_scores_layer_2 = []

    for eps in tqdm(eps_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels_0 = dbscan.fit_predict(list_of_embeddings[0])
        labels_1 = dbscan.fit_predict(list_of_embeddings[1])
        labels_2 = dbscan.fit_predict(list_of_embeddings[2])
        # check if DBSCAN found at least 2 clusters (silhouette_score requires at least 2 clusters)
        if len(set(labels_0)) > 1:
            silhouette_scores_layer_0.append(silhouette_score(list_of_embeddings[0], labels_0))
        else:
            silhouette_scores_layer_0.append(0)
        if len(set(labels_1)) > 1:
            silhouette_scores_layer_1.append(silhouette_score(list_of_embeddings[1], labels_1))
        else:
            silhouette_scores_layer_1.append(0)
        if len(set(labels_2)) > 1:
            silhouette_scores_layer_2.append(silhouette_score(list_of_embeddings[2], labels_2))
        else:
            silhouette_scores_layer_2.append(0)

    sns.lineplot(x=eps_values, y=silhouette_scores_layer_0, label='Layer 0', marker='o')
    sns.lineplot(x=eps_values, y=silhouette_scores_layer_1, label='Layer 1', marker='o')
    sns.lineplot(x=eps_values, y=silhouette_scores_layer_2, label='Layer 2', marker='o')
    plt.xlabel('DBSCAN eps value')
    plt.xticks(eps_values)
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Each Layer (DBSCAN)')
    plt.legend()
    plt.show()

def plot_correlation_heatmap(list_of_frame_colors, labels, layer_labels=None, figsize=(12, 4), cmap="viridis"):
    """
    Plots a heatmap of correlation between cluster colors and true labels for multiple layers.

    Args:
        list_of_frame_colors: list of np.ndarray, each shape (num_frames, 3 or 4), cluster colors per frame
        labels: np.ndarray of shape (num_frames,), true label per frame
        layer_labels: optional list of layer names
        figsize: figure size
        cmap: colormap for heatmap
    """
    num_layers = len(list_of_frame_colors)
    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(num_layers)]

    n_true_labels = labels.shape[0] 
    unique_labels = np.arange(n_true_labels)

    labels = labels.T


    plt.figure(figsize=figsize)
    for i, frame_colors in enumerate(list_of_frame_colors):
        # one hot encode cluster colors
        unique_colors, cluster_ids = np.unique(frame_colors, axis=0, return_inverse=True)
        n_clusters = len(unique_colors)
        print(f"Layer {i}: {n_clusters} unique clusters")
        one_hot = np.zeros((len(frame_colors), n_clusters))
        for j in range(n_clusters):
            one_hot[:, j] = (cluster_ids == j).astype(float)

        print(f"one_hot shape: {one_hot.shape}, label shape: {labels.shape}")
        # Compute correlations: rows are clusters, columns are labels
        corr_matrix = np.zeros((n_clusters, n_true_labels))
        for cluster_idx in range(n_clusters):
            for label_idx in range(n_true_labels):
                corr_matrix[cluster_idx, label_idx], _ = spearmanr(one_hot[:, cluster_idx], labels[:, label_idx])
        
        ax = plt.subplot(1, num_layers, i + 1)
        sns.heatmap(corr_matrix, ax=ax, cmap=cmap, cbar=(i==num_layers-1))
        ax.set_xlabel("True Label")
        ax.set_ylabel("Cluster ID")
       # ax.set_xticklabels([f"Label {l}" for l in unique_labels])
        ax.set_title(layer_labels[i])
    plt.tight_layout()
    plt.suptitle("Correlation Between Cluster Colors and True Labels", y=1.02)
    plt.show()



def compute_trajectory_embeddings(embeddings, window_size=30, stride=5, mean=False):
    """
    Compute trajectory embeddings by stacking consecutive frame embeddings within a sliding window.
    Args:
        embeddings: np.ndarray of shape (num_frames, embedding_dim)
        window_size: number of frames in each trajectory
        stride: step size for sliding window
    Returns:
        trajectory_embeddings: np.ndarray of shape (num_trajectories, window_size * embedding_dim)
    """
    num_frames, embedding_dim = embeddings.shape
    trajectory_embeddings = []
    for i in range(0, num_frames - window_size + 1, stride):
        if mean:
            trajectory_embeddings.append(np.mean(embeddings[i:i+window_size], axis=0))
        else:
            trajectory_embeddings.append(embeddings[i:i+window_size].flatten())

    return np.array(trajectory_embeddings)


def linear_plot_layer_embeddings(list_of_embeddings, dim=0, distance=False, angle=False, figsize=(12, 4)):
    """
    Plots the specified dimension of embedding for each frame for each layer as a line plot. 
    Additionally plot the distance/angle in latent space wandered from frame to frame.

    Args:
        list_of_embeddings: list of np.ndarray, each shape (num_frames, embedding_dim)
        dim: dimension of embedding to plot
        distance: whether to plot distance
        angle: whether to plot angle
        figsize: size of the figure
    """
    n_subplots = 1 + int(distance) + int(angle)
    fig, axs = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    axs = np.atleast_1d(axs)

    main_ax = axs[0]
    next_ax_idx = 1
    dist_ax = axs[next_ax_idx] if distance else None
    if distance:
        next_ax_idx += 1
    angle_ax = axs[next_ax_idx] if angle else None

    for i, embeddings in enumerate(list_of_embeddings):
        main_ax.plot(embeddings[:, dim] + i * 50, label=f"Layer {i} (+{i*50} offset)")

        if distance:
            dist = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            dist_ax.plot(dist + i * 20, label=f"Layer {i} Distance (+{i*20} offset)")

        if angle:
            dot_product = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
            norm_product = np.linalg.norm(embeddings[:-1], axis=1) * np.linalg.norm(embeddings[1:], axis=1)
            angle_vals = np.arccos(np.clip(dot_product / (norm_product + 1e-8), -1.0, 1.0))
            angle_ax.plot(angle_vals + i * 10, label=f"Layer {i} Angle (+{i*10} offset)")

    main_ax.set_ylabel(f"Embedding Dim {dim}")
    main_ax.set_title(f"Dimension {dim} of Embeddings per Frame")
    main_ax.legend()

    if distance:
        dist_ax.set_ylabel("L2 Distance")
        dist_ax.set_title("Frame-to-Frame Embedding Distance")
        dist_ax.legend()

    if angle:
        angle_ax.set_ylabel("Angle (rad)")
        angle_ax.set_title("Frame-to-Frame Embedding Angle")
        angle_ax.legend()

    axs[-1].set_xlabel("Frame index")
    plt.tight_layout()
    plt.show()

def fft_layer_embeddings(list_of_embeddings, dim=0, logscale=False):
    """
    Plots the FFT of the specified dimension of embedding for each layer.

    Args:
        list_of_embeddings: list of np.ndarray, each shape (num_frames, embedding_dim)
    """
    plt.figure(figsize=(12, 4))
    for i, embeddings in enumerate(list_of_embeddings):
        fft_vals = np.fft.fft(embeddings[:, dim])
        if logscale:
            fft_vals = np.log(np.abs(fft_vals) + 1e-8)
        freqs = np.fft.fftfreq(len(embeddings))
        plt.plot(freqs[:len(freqs)//2], np.abs(fft_vals)[:len(fft_vals)//2], label=f"Layer {i}")
    plt.xlabel("Frequency")
    plt.ylabel(f"FFT Magnitude of Embedding Dim {dim}")
    plt.title(f"FFT of Dimension {dim} of Embeddings")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_speed(keypoints, center_idx):
    """Compute instant speed of the center keypoint from keypoints array."""

    center_data = keypoints[:, center_idx] 

    # compute euclidean distance between consecutive frames for the center keypoint
    diff = np.linalg.norm(np.diff(center_data, axis=0), axis=1)

    # starting speed is 0
    diff = np.insert(diff, 0, 0)

    # interpolate nans if any
    if np.isnan(diff).any():
        nans = np.isnan(diff)
        not_nans = ~nans
        diff[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), diff[not_nans])

    return diff