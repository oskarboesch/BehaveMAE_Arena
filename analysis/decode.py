import torch
import sys
# add the project root to the Python path
sys.path.append("/home/boesch/BehaveMAE/")
from util.misc import load_model  # adjust import path as needed
from datasets.pose_traj_dataset import BasePoseTrajDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def decode(checkpoint_path, latents, keypoints=None):
    from models.models_defs import hbehavemae
    from models.hbehave_mae import apply_fusion_head

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load args from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    args.resume = checkpoint_path  # tell load_model where to find the checkpoint
    args.eval = True
    model = hbehavemae(
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoding_strategy=args.decoding_strategy,
        norm_loss=args.norm_loss,
        q_strides=args.q_strides,
        patch_kernel=args.patch_kernel,
        init_embed_dim=args.init_embed_dim,
        out_embed_dims=args.out_embed_dims,
        stages=args.stages,
        mask_unit_attn=args.mask_unit_attn,
        init_num_heads=args.init_num_heads,
    )
    load_model(args, model, optimizer=None, loss_scaler=None)
    model = model.to(device).eval()

    mask_unit_shape = model.mask_unit_size
    print(f"Using mask unit shape from model config: {mask_unit_shape}")

    run_id = list(list(latents.values())[0].keys())[0]  # Get a sample run_id from the latents dict
    print(f"Example latent shape for run {run_id}:")
    keypoints = keypoints[run_id]
    keypoints = keypoints.reshape(-1, keypoints.shape[1]*keypoints.shape[2]) 
    keypoints = BasePoseTrajDataset._normalize(keypoints, grid_size=500)
    latents = [torch.from_numpy(latents[run_id]).to(device) for latents in latents.values()]
    mask = torch.ones((1, latents[0].shape[0] // mask_unit_shape[0]), device=device)  # Batch size 1, sequence length adjusted for mask unit size

    print(f"Mask shape: {mask.shape}")
    # we need to check length to match mask unit size
    mu_shapes = [model.mask_unit_size]  # stage 0: [75, 1, 1]
    curr = list(model.mask_unit_size)
    for ix in range(model.q_pool):
        curr = [i // s for i, s in zip(curr, model.q_strides[ix])]
        mu_shapes.append(curr)  # stage 1: [25,1,1], stage 2: [1,1,1] etc.

    latents = [
        latent.view(1, -1, *mu_shape, latent.shape[1])
        for latent, mu_shape in zip(latents, mu_shapes)
    ]
    for i, latent in enumerate(latents):
        print(f"Latent {i} shape before padding: {latent.shape}")

    with torch.no_grad():
        latents = latents[: model.q_pool] + latents[-1:]

        if model.decoding_strategy == "single":
            # Use only the last layer's output for decoding
            x = latents[-1]

        else:
            # Multi-scale fusion
            x = 0.0
            for head, interm_x in zip(model.multi_scale_fusion_heads, latents):
                x += apply_fusion_head(head, interm_x)

        x = model.encoder_norm(x)

        pred, pred_mask = model.forward_decoder(x, mask)
        print("pred shape:", pred.shape)
        print("pred_mask shape:", pred_mask.shape)
        keypoints_preds = model.unpatch_label_3d(pred, pred_mask)
        print("Decoded keypoints shape:", keypoints_preds.shape)

    # plot one frame of the decoded keypoints
    import matplotlib.pyplot as plt
    import numpy as np

    keypoints_preds = keypoints_preds.cpu().numpy().reshape(keypoints_preds.shape[0], keypoints_preds.shape[2] // 2, 2)
    keypoints = keypoints.reshape(keypoints.shape[0], keypoints.shape[1] // 2, 2)

    sampled_preds = keypoints_preds[:10].reshape(-1, 2)
    sampled_true  = keypoints[:10].reshape(-1, 2)
    n_pred = len(sampled_preds)
    n_true = len(sampled_true)

    fig, ax = plt.subplots(figsize=(8, 8))

    sc_true = ax.scatter(
        sampled_true[:, 0], sampled_true[:, 1],
        c=np.arange(n_true), cmap="viridis", alpha=0.6, s=15, label="Original"
    )
    sc_pred = ax.scatter(
        sampled_preds[:, 0], sampled_preds[:, 1],
        c=np.arange(n_pred), cmap="magma", alpha=0.6, s=15, label="Predicted"
    )

    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.1)
    cax2 = divider.append_axes("right", size="5%", pad=0.7)

    fig.colorbar(sc_true, cax=cax1, label="Frame index (Original)")
    fig.colorbar(sc_pred, cax=cax2, label="Frame index (Predicted)")

    ax.set_title(f"Decoded vs Original Keypoints — run {run_id} - first 10 frames")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig("decoded_keypoints_vs_original.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":

    # Example usage
    checkpoint_path = "/scratch/izar/boesch/BehaveMAE/outputs/arena/experiment1/checkpoint-00199.pth"
    embeddings_path = "/scratch/izar/boesch/BehaveMAE/outputs/arena/experiment1/embeddings.npy"
    keypoints_path = "/scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz"

    keypoints = np.load(keypoints_path, allow_pickle=True)['keypoints'].item()  # Assuming keypoints are stored under 'keypoints' key
    latents = np.load(embeddings_path, allow_pickle=True).item()  # Assuming embeddings are saved as a dict of layer_key -> tensor
    decode(checkpoint_path, latents, keypoints)