import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from models.hbehave_mae import HBehaveMAE

def plot_pos_embed(model, t_tokens: int, cyclic: bool = False):
    """Visualize positional embeddings for a given number of temporal tokens."""
    with torch.no_grad():
        if cyclic:
            pos_embed = model.get_pos_embed_cyclic(t_tokens)  # (1, T*H*W, D)
        else:
            pos_embed = model.get_pos_embed(t_tokens)  # (1, T*H*W, D)

    pos_embed = pos_embed.squeeze(0).cpu().numpy()  # (N_tokens, D)
    N, D = pos_embed.shape
    H, W = model.tokens_spatial_shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Positional Embeddings — T_tokens={t_tokens}, spatial={H}×{W}, total={N} tokens", fontsize=13)

    # ── 1. Raw embedding matrix (tokens × dims) ────────────────────────────
    ax = axes[0]
    im = ax.imshow(pos_embed, aspect="auto", cmap="RdBu_r")
    ax.set_title("Embedding matrix\n(tokens × dims)")
    ax.set_xlabel("Embedding dim")
    ax.set_ylabel("Token index")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    # ── 2. Cosine similarity matrix (tokens × tokens) ──────────────────────
    ax = axes[1]
    norm = pos_embed / (np.linalg.norm(pos_embed, axis=-1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    im = ax.imshow(sim, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    ax.set_title("Cosine similarity\n(tokens × tokens)")
    ax.set_xlabel("Token index")
    ax.set_ylabel("Token index")
    # draw grid lines separating temporal groups
    spatial_tokens = H * W
    for t in range(1, t_tokens):
        ax.axhline(t * spatial_tokens - 0.5, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(t * spatial_tokens - 0.5, color="white", linewidth=0.5, alpha=0.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    # ── 3. Mean embedding per temporal step (spatial average) ──────────────
    ax = axes[2]
    temporal_mean = pos_embed.reshape(t_tokens, H * W, D).mean(axis=1)  # (T, D)
    im = ax.imshow(temporal_mean, aspect="auto", cmap="RdBu_r")
    ax.set_title("Mean embedding per temporal step\n(T_tokens × dims)")
    ax.set_xlabel("Embedding dim")
    ax.set_ylabel("Temporal token index")
    ax.set_yticks(np.arange(t_tokens))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(f"pos_embed_T{t_tokens}-{'cyclic' if cyclic else 'linear'}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved pos_embed_T{t_tokens}-{'cyclic' if cyclic else 'linear'}.png  |  embed shape: {pos_embed.shape}")


# Test autonome
if __name__ == "__main__":
    t_tokens = 20
    # Paramètres minimaux pour instancier HBehaveMAE
    model = HBehaveMAE(
        in_chans=1,
        patch_stride=(2, 1, 3),
        mlp_ratio=4.0,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=1,
        decoding_strategy="single"
    )
    plot_pos_embed(model, t_tokens=t_tokens, cyclic = True)