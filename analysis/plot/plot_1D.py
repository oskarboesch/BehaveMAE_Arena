import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns


def plot_1D(
    embeddings: dict,           # {layer_key: {run_id: array (T_tokens, D)}}
    token_shapes: dict,         # {layer_idx: (T_tokens, ...)} or {layer_idx: int}
    kinematics: dict = None,    # {run_id: {kin_var: array (T_frames,)}}
    dims: int = 8,
    title: str = None,
    figsize: tuple = (12, 3),   # per-row figsize
    output_path: str = None,
    random_state: int = 42,
):
    """
    For the first run_id found in embeddings, plots a heatmap of the first `dims`
    dimensions over time for each layer, upsampled to the original frame rate using
    token_shapes. If kinematics is provided, appends kinematic traces below,
    all sharing the same time axis (original frames).

    Layout:
        - one row per layer  (heatmap: original_frames × dims)
        - one row per kinematic variable (line plot), if provided
    """
    np.random.seed(random_state)
    print("Plotting")

    layer_keys = list(embeddings.keys())
    first_run_id = list(embeddings[layer_keys[0]].keys())[0]

    # Infer original frame length from kinematics or the finest layer
    if kinematics is not None and first_run_id in kinematics:
        kin_vars = list(kinematics[first_run_id].keys())[:4]
        T_original = len(np.array(kinematics[first_run_id][kin_vars[0]]))
    else:
        kin_vars = []
        # fall back: use the layer with the most tokens as reference
        max_tokens = max(
            np.array(embeddings[lk][first_run_id]).shape[0] for lk in layer_keys
        )
        T_original = max_tokens

    n_rows = len(layer_keys) + len(kin_vars)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(figsize[0], figsize[1] * n_rows),
        sharex=True,                # share x so layers + kinematics are aligned
    )
    if n_rows == 1:
        axes = [axes]

    # ── embedding heatmaps ────────────────────────────────────────────────────
    for row_idx, layer_key in enumerate(layer_keys):
        ax = axes[row_idx]

        layer_idx = int(layer_key.split("_")[-1])
        emb = np.array(embeddings[layer_key][first_run_id])   # (T_tokens, D)
        T_tokens, D = emb.shape
        emb_plot = emb[:, :dims]                              # (T_tokens, min(dims,D))

        # ── upsample to original frame rate ──────────────────────────────────
        # token_shapes[layer_idx] gives the token grid shape for this layer,
        # e.g. (T_tokens,) or (T_tokens, K) — we only need T_tokens here.
        token_shape = token_shapes[layer_idx]
        # each token covers this many original frames
        frames_per_token = token_shape[0]     
        # repeat each token's row to fill its original frame span
        repeat_counts = np.diff(
            np.round(np.arange(T_tokens + 1) * frames_per_token).astype(int)
        )                                                       # (n_tokens,)
        emb_upsampled = np.repeat(emb_plot, repeat_counts, axis=0)  # (T_original, dims)
        # imshow: extent sets the real x-axis range [0, T_original]
        # so sharex works correctly across all subplots
        vmax = np.percentile(np.abs(emb_upsampled), 95)        # robust color scale
        im = ax.imshow(
            emb_upsampled.T,                                    # (dims, T_original)
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[0, T_original, 0, min(dims, D)],           # x: frames, y: dims
            origin="lower",
            interpolation="nearest",
        )
        ax.set_ylabel(f"{layer_key}\n(dim)", fontsize=9)
        ax.set_title(
            f"{layer_key} — first {min(dims, D)} dims  "
            f"({T_tokens} tokens → {T_original} frames, "
            f"{frames_per_token:.1f} frames/token)",
            fontsize=8,
        )
        ax.set_xlim(0, T_original)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1.5%", pad=0.05)
        plt.colorbar(im, cax=cax, label="activation")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))


    # ── kinematic line plots ──────────────────────────────────────────────────
    for kin_idx, kin_var in enumerate(kin_vars):
        ax = axes[len(layer_keys) + kin_idx]
        kin_trace = np.array(kinematics[first_run_id][kin_var])   # (T_original,)
        # pad at the end if needed (in case kinematics is shorter than T_original)
        if len(kin_trace) < T_original:
            kin_trace = np.pad(kin_trace, (0, T_original - len(kin_trace)), mode="edge")
        x_frames = np.arange(T_original)
        ax.plot(x_frames, kin_trace, linewidth=0.8, color="steelblue")
        ax.set_ylabel(kin_var, fontsize=9)
        ax.set_xlim(0, T_original - 1)
        ax.set_xlim(0, T_original)  
        divider = make_axes_locatable(ax)
        cax_dummy = divider.append_axes("right", size="1.5%", pad=0.05)
        cax_dummy.set_visible(False)

        sns.despine(ax=ax)

    axes[-1].set_xlabel("frames (original rate)", fontsize=9)
    for ax in axes:
        ax.set_xlim(0, T_original)

    # ── global title & save ───────────────────────────────────────────────────
    fig.suptitle(
        title if title else f"Embedding activations across layers — run {first_run_id}",
        fontsize=11,
        y=1.001,
    )
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()