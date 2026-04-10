import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.cm as cm
from datasets.arena_dataset import ArenaDataset
from datasets.arena_dataset import get_kp_colors
from tqdm import tqdm

def embeds_to_pose(embeddings, clustroids_idxs, layer_window_maps, window_size, token_shapes, keypoints, output_dir, data_type="raw"):
    def convert_window_to_frame(run_id, window_idx, layer_window_map, token_shape):
        return (run_id, int(layer_window_map[run_id][window_idx] * token_shape[0] + token_shape[0] // 2))
    
    clustroids_center_frames = {}
    for layer_key, alg_dict in clustroids_idxs.items():
        clustroids_center_frames[layer_key] = {}
        token_shape = token_shapes[int(layer_key.split("_")[-1])]
        for alg_name, cluster_indices in alg_dict.items():
            clustroids_center_frames[layer_key][alg_name] = [
                {
                    'clustroid': convert_window_to_frame(*cluster_info['clustroid'], layer_window_maps[layer_key], token_shape),
                    'members': [convert_window_to_frame(*m, layer_window_maps[layer_key], token_shape) for m in cluster_info['members']]
                }
                for cluster_info in cluster_indices
            ]

    output_dir = os.path.join(output_dir, data_type)
    _plot_keypoint_trajectories(clustroids_center_frames, keypoints, window_size, token_shapes, output_dir)
    _plot_position_heatmaps(clustroids_center_frames, keypoints, window_size, token_shapes, output_dir)

def _plot_keypoint_trajectories(clustroids_center_frames, keypoints, window_size, token_shapes, output_dir, max_plots_per_alg=50):
    """Plot typical keypoint trajectories (median of aligned cluster members) with increasing alpha over time."""
    if keypoints is None:
        print("No keypoints provided, skipping embeds_to_pose.")
        return
    
    MAX_RES = 10
    
    skeleton = ArenaDataset.get_skeleton()
    os.makedirs(output_dir, exist_ok=True)
    
    for layer_key, alg_dict in clustroids_center_frames.items():
        token_shape = max(window_size, token_shapes[int(layer_key.split("_")[-1])][0])
        is_centroid = token_shape >=100
        print(f"Processing {layer_key} with token shape {token_shape}. Centroid-based: {is_centroid}")
        half_window = token_shape // 2
        seq_length = 2 * half_window + 1
        indices_to_keep = [ArenaDataset.BODY_PART_2_INDEX[name] for name in ArenaDataset.SUBSAMPLED_KEYPOINTS]

        for alg_name, center_frames in alg_dict.items():
            save_dir = os.path.join(output_dir, "figures", layer_key, alg_name, "trajectory_plots")
            os.makedirs(save_dir, exist_ok=True)
            list_of_seqs_for_dendrogram = []
            for idx, cluster_info in tqdm(enumerate(center_frames[:max_plots_per_alg]), desc=f"Processing {layer_key} | {alg_name}"):
                if is_centroid:
                    # select simply the centroid from the cluster_info['clustroid'] as the typical sequence, without alignment or median since the clustroid is already the most central point in the cluster
                    m_run_id, m_center = cluster_info['clustroid']
                    start_frame = m_center - half_window
                    end_frame = m_center + half_window + 1
                    run_kpts = keypoints[m_run_id][start_frame:end_frame, indices_to_keep, :]
                    if run_kpts.size == 0 or np.all(~np.isfinite(run_kpts)):
                        print(f"Centroid sequence for cluster {idx} in {layer_key} | {alg_name} is empty or all NaNs, skipping.")
                        continue
                    run_kpts = ArenaDataset.interpolate_nans(run_kpts).reshape(len(run_kpts), -1, 2)  # (n_frames, n_kpts, 2)
                    typical_seq = run_kpts.copy()  # (seq_length, num_kpts, 2)
                else:
                    member_seqs = []
                    MAX_MEMBERS = 200
                    members = cluster_info['members']
                    if len(members) > MAX_MEMBERS:
                        rng = np.random.default_rng(42)
                        indices = rng.choice(len(members), size=MAX_MEMBERS, replace=False)
                        members = [members[i] for i in indices]  # list indexing preserves tuple types
                    for m_run_id, m_center in members:
                        if m_run_id not in keypoints:
                            continue
                        run_kpts = keypoints[m_run_id][:, indices_to_keep, :]
                        start_frame = m_center - half_window
                        end_frame = m_center + half_window + 1

                        if start_frame >= 0 and end_frame <= len(run_kpts):
                            # Shape: (seq_length, num_kpts, 2_or_3)
                            seq = run_kpts[start_frame:end_frame].copy()  
                            seq = ArenaDataset.interpolate_nans(seq).reshape(len(seq), -1, 2)  # (n_frames, num_kpts, 2)        

                            centroid = np.nanmean(seq[0], axis=0) 
                            seq_aligned = seq - centroid
                            seq_aligned = _align_sequence_to_heading(seq_aligned, nose_idx=1, tail_idx=7)  # align to body axis
                            
                            member_seqs.append(seq_aligned)

                    if len(member_seqs) == 0:
                        print(f"No valid members for cluster {idx} in {layer_key} | {alg_name}, skipping trajectory plot.")
                        continue
                    stacked_seqs = np.stack(member_seqs, axis=0) 
                    typical_seq = np.nanmedian(stacked_seqs, axis=0) 
                sampling_rate = min(25, max(1, seq_length // MAX_RES))  # Cap to avoid too sparse sampling
                typical_seq = typical_seq[::sampling_rate]  # (n_sampled_frames, num_kpts, 2_or_3)
                kp_colors = get_kp_colors(subsampled=True)
                save_path = os.path.join(save_dir, f"cluster_{idx}_trajectory.png")
                _plot_keypoint_trajectory(typical_seq, skeleton, kp_colors, save_path, is_centroid=is_centroid)
                list_of_seqs_for_dendrogram.append(typical_seq)
            print(f"Saved typical pose plots at {save_dir}")
            dendro_save_path = os.path.join(save_dir, f"cluster_trajectories_dendrogram.png")
            plot_similarity_dendrogram(list_of_seqs_for_dendrogram, dendro_save_path, target_length=MAX_RES)
            print(f"Saved trajectory similarity dendrogram to {dendro_save_path}")

def _plot_keypoint_trajectory(seq, skeleton, kp_colors, save_path, is_centroid):
    import matplotlib.colors as mc
    
    n_frames = len(seq)
    alpha_power = 1.0 if is_centroid else 2.0
    dot_size = 50 if is_centroid else 400
    alphas = np.linspace(0.2, 1.0, n_frames) ** alpha_power

    fig, ax = plt.subplots(figsize=(6, 6))

    # Fonction pour fondre une couleur avec le blanc (simule l'alpha en 100% opaque)
    def blend_with_white(color, a):
        return tuple(a * val + (1 - a) * 1.0 for val in mc.to_rgb(color))

    for t, kpts_t in enumerate(seq):
        alpha = float(alphas[t])
        size = dot_size
        lw = 1.0 if t == n_frames - 1 else 0.5
        edgecolor = blend_with_white("black", max(0, alpha - 0.05))

        # Draw skeleton FIRST (behind dots) at same zorder level
        if skeleton is not None:
            for (i, j) in skeleton:
                if i < len(kpts_t) and j < len(kpts_t):
                    # Ligne de contour (dessinée en gris simulé opaque)
                    ax.plot([kpts_t[i, 0], kpts_t[j, 0]],
                            [kpts_t[i, 1], kpts_t[j, 1]],
                            color=blend_with_white("black", max(0, alpha - 0.05)), 
                            alpha=1.0, # 100% opaque
                            linewidth=7*dot_size/400, zorder=t * 3)

                    # Ligne interne (dessinée en couleur pastel simulée opaque)
                    ax.plot([kpts_t[i, 0], kpts_t[j, 0]],
                            [kpts_t[i, 1], kpts_t[j, 1]],
                            color=blend_with_white(kp_colors[i], alpha), 
                            alpha=1.0, # 100% opaque
                            linewidth=5*dot_size/400, zorder=t * 3 + 1)

        # Draw dots on top of skeleton, same time step
        for k, (x, y) in enumerate(kpts_t[:, :2]):
            c_dot = blend_with_white(kp_colors[k], alpha)
            ax.scatter(x, y, s=size, color=c_dot,
                    edgecolors=edgecolor, linewidths=lw,
                    alpha=1.0, zorder=t * 3 + 2)          # 100% opaque
            
    ax.set_title(f"Typical Pose Trajectory (n={n_frames} frames)", pad=20)
    if is_centroid:
        # Draw arena circle (diameter 500 → radius 250, centered at 250, 250)
        arena = plt.Circle((250, 250), 250, fill=False, edgecolor="gray", linewidth=1.5, linestyle="--", zorder=0)
        ax.add_patch(arena)
        ax.set_xlim(-25, 525)
        ax.set_ylim(-25, 525)
    else:
        ax.axis("off")
        all_plotted_points = seq.reshape(-1, seq.shape[-1])
        if len(all_plotted_points) == 0 or np.all(np.isnan(all_plotted_points[:, :2])):
            plt.close(fig)
            return
        x_min, y_min = np.nanmin(all_plotted_points[:, :2], axis=0)
        x_max, y_max = np.nanmax(all_plotted_points[:, :2], axis=0)
        pad_x = max((x_max - x_min) * 0.3, 1.0)
        pad_y = max((y_max - y_min) * 0.3, 1.0)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    ax.set_aspect('equal', adjustable='box') # Garder les proportions réelles
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    


def _plot_position_heatmaps(cluster_center_frames, keypoints, window_size, token_shapes, output_dir,
                             grid_size=100, max_plots_per_alg=50):
    if keypoints is None:
        print("No keypoints provided, skipping position heatmaps.")
        return
    geom_by_stem = load_arena_geometries()
    os.makedirs(output_dir, exist_ok=True)

    for layer_key, alg_dict in cluster_center_frames.items():
        token_shape = token_shapes[int(layer_key.split("_")[-1])]
        half_window = max(window_size, token_shape[0]) // 2

        for alg_name, center_frames in alg_dict.items():
            save_dir = os.path.join(output_dir, "figures", layer_key, alg_name, "trajectory_plots")
            os.makedirs(save_dir, exist_ok=True)

            n_clusters = min(len(center_frames), max_plots_per_alg)

            # --- Step 1: build all heatmaps first ---
            heatmaps = []
            n_members_list = []

            for cluster_info in center_frames[:max_plots_per_alg]:
                heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
                for run_id, center_frame in cluster_info['members']:
                    if run_id not in keypoints:
                        continue
                    if run_id not in geom_by_stem.index:
                        continue
                    run_kpts = keypoints[run_id]
                    n_run_frames = len(run_kpts)
                    start_frame = max(0, center_frame - half_window)
                    end_frame = min(n_run_frames, center_frame + half_window + 1)
                    if start_frame >= end_frame:
                        continue
                    pose_seq = run_kpts[start_frame:end_frame]  # (n_frames, n_kpts, 2)
                    positions = pose_seq[:, 9, :].copy()        # (n_frames, 2)
                    cx, cy, radius = get_arena_geom_for_run(run_id, geom_by_stem)
                    x_n, y_n, _ = normalize_run_xy(positions[:, 0], positions[:, 1], cx, cy, radius)
                    valid_mask = np.isfinite(x_n) & np.isfinite(y_n)
                    if not np.any(valid_mask):
                        continue
                    x_n = x_n[valid_mask]
                    y_n = y_n[valid_mask]
                    xi = np.clip(((x_n + 1.05) / 2.1 * grid_size).astype(int), 0, grid_size - 1)
                    yi = np.clip(((y_n + 1.05) / 2.1 * grid_size).astype(int), 0, grid_size - 1)
                    np.add.at(heatmap, (yi, xi), 1)

                total = heatmap.sum()
                if total > 0:
                    import scipy.ndimage
                    heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=2.0)
                    heatmap /= heatmap.sum()
                heatmaps.append(heatmap)
                n_members_list.append(len(cluster_info['members']))

            if len(heatmaps) == 0:
                print(f"No valid heatmaps for {layer_key} | {alg_name}, skipping.")
                continue

            # --- Step 2: shared norm across all clusters ---
            display_maps = [map_for_display(h) for h in heatmaps]
            norm, vmin, vmax, norm_desc = get_color_norm(display_maps)

            # --- Step 3: plot ---
            fig, axes = plt.subplots(1, n_clusters, figsize=(4.8 * n_clusters, 5))
            if n_clusters == 1:
                axes = [axes]

            mappable = None
            for idx, (ax, heatmap, n_members) in enumerate(zip(axes, heatmaps, n_members_list)):
                H_disp = map_for_display(heatmap)
                mappable = ax.imshow(
                    H_disp, origin='lower', extent=[-1.05, 1.05, -1.05, 1.05],
                    cmap='magma', norm=norm,
                )
                ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='white', lw=1.2))
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"Cluster {idx}\n(n={n_members})", fontsize=10)

            fig.suptitle(
                f"{layer_key} | {alg_name} — position heatmaps"
                f"\n(cmap=magma, vmin={vmin:.3e}, vmax={vmax:.3e})",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 0.92, 0.92])

            # Add colorbar after tight_layout to avoid resetting its position
            pos = axes[-1].get_position()
            cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
            cbar = fig.colorbar(mappable, cax=cax)
            cbar.set_label(f'Occupancy probability | norm={norm_desc}')

            save_path = os.path.join(save_dir, f"heatmap_{alg_name}.png")
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
        print(f"Saved heatmap plots for {layer_key} at {save_dir}")


def get_arena_geom_for_run(run_id, geom_by_stem):
    row = geom_by_stem.loc[run_id]
    return float(row['center_x']), float(row['center_y']), float(row['radius'])

def normalize_run_xy(x, y, cx, cy, radius):
    x_n = (x - cx) / radius
    y_n = (y - cy) / radius
    d_n = np.sqrt(x_n ** 2 + y_n ** 2)
    return x_n, y_n, d_n

def load_arena_geometries():
    import pandas as pd
    geom_path = "/scratch/izar/boesch/data/Arena_Data/openfield_ORT_field_centers.csv"
    geom_df = pd.read_csv(geom_path)
    # video_file name are run_id.mpg
    geom_by_stem = geom_df.set_index(geom_df['video_filename'].apply(lambda x: x.split('.')[0]))
    return geom_by_stem

def map_for_display(H, use_log=False, eps=1e-6):
    H = np.asarray(H, dtype=float)
    H = np.where(np.isfinite(H), H, 0.0)
    if use_log:
        return np.log10(np.clip(H, eps, None))
    return H


def get_color_norm(display_maps, use_log=False):
    vals = np.concatenate([m.ravel() for m in display_maps])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        vals = np.array([0.0, 1.0], dtype=float)

    mode = str("magma").strip().lower()
    q_low = float(np.clip(0.05, 0.0, 1.0))
    q_high = float(np.clip(0.95, 0.0, 1.0))
    if q_high <= q_low:
        q_high = min(1.0, q_low + 1e-6)

    if mode == 'quantile':
        vmin = float(np.nanquantile(vals, q_low))
        vmax = float(np.nanquantile(vals, q_high))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm_desc = f'quantile[{q_low:.3f}, {q_high:.3f}]'
    elif mode == 'power' and not use_log:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        norm = mcolors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        norm_desc = f'power(gamma={0.5:.1f})'
    elif mode == 'log' and not use_log:
        positive = vals[vals > 0]
        if positive.size == 0:
            vmin, vmax = 1e-12, 1.0
        else:
            vmin = float(np.nanquantile(positive, max(q_low, 0.001)))
            vmax = float(np.nanquantile(positive, q_high))
        vmin = max(vmin, 1e-12)
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        norm_desc = f'log[{q_low:.3f}, {q_high:.3f}]'
    else:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm_desc = 'linear'

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm_desc = 'linear(fallback)'

    return norm, vmin, vmax, norm_desc


def test_plot_trajectory():
    # Simulate a typical trajectory with 10 keypoints over 21 frames
    keypoints_path = "/scratch/izar/boesch/data/Arena_Data/shuffle-3_split-test.npz"
    keypoints = np.load(keypoints_path, allow_pickle=True)['keypoints'].item()
    seq = keypoints[list(keypoints.keys())[0]][:11] 
    center = len(seq) // 2
    indices_to_keep = [ArenaDataset.BODY_PART_2_INDEX[name] for name in ArenaDataset.SUBSAMPLED_KEYPOINTS]
    seq = seq[:, indices_to_keep]
    skeleton = ArenaDataset.get_skeleton()
    kp_colors = get_kp_colors(subsampled=True)
    save_path = "test_trajectory.png"
    _plot_keypoint_trajectory(seq, skeleton, kp_colors, save_path)
    print(f"Test trajectory plot saved to {save_path}")

def _align_sequence_to_heading(seq, nose_idx, tail_idx):
    """Rotate seq so the body axis at center frame points right (angle=0)."""
    nose = seq[0, nose_idx, :2]
    tail = seq[0, tail_idx, :2]
    
    body_vec = nose - tail
    angle = np.arctan2(body_vec[1], body_vec[0])  # current heading angle
    
    # Rotation matrix to bring angle -> 0
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a],
                  [ sin_a,  cos_a]])
    
    # Apply rotation to all frames and keypoints
    seq_rotated = seq.copy()
    seq_rotated[..., :2] = (R @ seq[..., :2].reshape(-1, 2).T).T.reshape(seq.shape[:-1] + (2,))
    return seq_rotated

def plot_similarity_dendrogram(seq_list, save_path, target_length=None):
    """Plot a dendrogram of similarity between cluster sequences"""
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    if len(seq_list) < 2:
        print("Not enough sequences for dendrogram, skipping.")
        return

    # Flatten sequences to (n_samples, n_features)
    flattened = [seq.reshape(-1) for seq in seq_list]
    if target_length is None:
        target_length = min(f.shape[0] for f in flattened)  # truncate to shortest
    normalized = []
    for f in flattened:
        if len(f) >= target_length:
            normalized.append(f[:target_length])
        else:
            # Pad with zeros (or np.nan then fill)
            pad = np.zeros(target_length)
            pad[:len(f)] = f
            normalized.append(pad)
    flattened = np.stack(normalized, axis=0)  # (n_seqs, target_length) — guaranteed 2D
    distance_matrix = pdist(flattened, metric='euclidean')
    Z = linkage(distance_matrix, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title("Dendrogram of Cluster Sequence Similarity")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved similarity dendrogram to {save_path}")
    plt.close()
if __name__ == "__main__":
    test_plot_trajectory()