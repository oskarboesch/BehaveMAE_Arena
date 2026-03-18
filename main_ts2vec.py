from ts2vec import TS2Vec
from datasets.arena_dataset import ArenaDataset
import numpy as np
from tqdm import tqdm
import argparse
import torch
import os


def get_args_parser():
    parser = argparse.ArgumentParser("TS2Vec pre-training on Arena", add_help=False)

    # Data
    parser.add_argument("--train_data_path", default="/scratch/izar/boesch/data/Arena_Data/shuffle-3_split-test.npz", type=str)
    parser.add_argument("--output_dir", default="/scratch/izar/boesch/ts2vec/outputs/arena", type=str)
    parser.add_argument("--num_frames", default=900, type=int, help="Number of frames per window")
    parser.add_argument("--num_keypoints", default=27, type=int)
    parser.add_argument("--kpts_dimensions", default=2, type=int)

    # Model
    parser.add_argument("--output_dims", default=128, type=int, help="Embedding dimension")
    parser.add_argument("--hidden_dims", default=64, type=int, help="Hidden dimension in encoder")
    parser.add_argument("--depth", default=10, type=int, help="Number of dilated conv layers")

    # Training
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_train_length", default=None, type=int, help="Max sequence length during training (for memory)")
    parser.add_argument("--n_train_samples", default=5, type=int, help="Number of windows to subsample for training (set to -1 to use all)")
    parser.add_argument("--device", default=0, type=int, help="GPU device id")
    parser.add_argument("--num_workers", default=8, type=int)

    # Inference / sliding window
    parser.add_argument("--sliding_length", default=1, type=int, help="Sliding inference step size")
    parser.add_argument("--sliding_padding", default=50, type=int, help="Context padding for sliding inference")

    return parser


def extract_train_data(dataset, n_samples, num_workers, batch_size):
    """Extract training windows into a numpy array, with optional subsampling."""
    n_total = len(dataset)
    if n_samples > 0 and n_samples < n_total:
        indices = np.random.choice(n_total, size=n_samples, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
    )

    all_data = []
    for batch, _ in tqdm(loader, desc="Extracting training windows"):
        # batch shape: (B, n_frames, 1, 54) → squeeze individual dim
        batch = batch.numpy().squeeze(2)  # (B, n_frames, 54)
        all_data.append(batch)

    train_data = np.concatenate(all_data, axis=0)  # (n_samples, n_frames, 54)
    print(f"Train data shape: {train_data.shape}  ({train_data.nbytes / 1e9:.2f} GB)")
    return train_data


def run_inference(model, infer_dataset, args):
    """Run per-sequence inference and return embedding dicts."""
    ts_level_embeddings = {}
    instance_level_embeddings = {}
    ts_level_sliding_embeddings = {}

    for sequence_name, sequence in tqdm(
        zip(infer_dataset.sequence_names, infer_dataset.sequences),
        total=len(infer_dataset.sequences),
        desc="Inference"
    ):
        # sequence shape: (n_frames, n_keypoints, kpts_dims) → flatten keypoints
        seq = sequence.reshape(1, sequence.shape[0], -1).astype(np.float32)  # (1, n_frames, 54)

        # Timestamp-level
        ts_emb = model.encode(seq)  # (1, n_frames, output_dims)
        ts_level_embeddings[sequence_name] = ts_emb[0]  # (n_frames, output_dims)

        # Instance-level
        inst_emb = model.encode(seq, encoding_window="full_series")  # (1, output_dims)
        instance_level_embeddings[sequence_name] = inst_emb[0]  # (output_dims,)

        # Sliding causal
        slide_emb = model.encode(
            seq,
            causal=True,
            sliding_length=args.sliding_length,
            sliding_padding=args.sliding_padding,
        )  # (1, n_frames, output_dims)
        ts_level_sliding_embeddings[sequence_name] = slide_emb[0]  # (n_frames, output_dims)

    return ts_level_embeddings, instance_level_embeddings, ts_level_sliding_embeddings


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_dims = args.num_keypoints * args.kpts_dimensions  # 54

    # Datasets
    print("Loading training dataset...")
    train_dataset = ArenaDataset(
        path_to_data_dir=args.train_data_path,
        mode="pretrain",
        num_frames=args.num_frames,
    )

    print("Loading inference dataset...")
    infer_dataset = ArenaDataset(
        path_to_data_dir=args.train_data_path,
        mode="inference",
    )

    # Extract training data (subsampled if needed)
    train_data = extract_train_data(
        train_dataset,
        n_samples=args.n_train_samples,
        num_workers=args.num_workers,
        batch_size=args.batch_size * 8,  # larger batch for extraction only
    )

    # Model
    model = TS2Vec(
        input_dims=input_dims,
        device=args.device,
        output_dims=args.output_dims,
        hidden_dims=args.hidden_dims,
        depth=args.depth,
        lr=args.lr,
        batch_size=args.batch_size,
        max_train_length=args.max_train_length,
    )

    # Training
    print("Training TS2Vec...")
    loss_log = model.fit(
        train_data,
        verbose=True,
        n_epochs=args.epochs,
    )
    np.save(os.path.join(args.output_dir, "loss_log.npy"), np.array(loss_log))
    model.save(os.path.join(args.output_dir, "model.pkl"))
    print(f"Model saved to {args.output_dir}")

    # Inference
    print("Running inference...")
    ts_embs, inst_embs, slide_embs = run_inference(model, infer_dataset, args)

    # Save embeddings
    np.save(os.path.join(args.output_dir, "ts_level_embeddings.npy"), ts_embs)
    np.save(os.path.join(args.output_dir, "instance_level_embeddings.npy"), inst_embs)
    np.save(os.path.join(args.output_dir, "ts_level_sliding_embeddings.npy"), slide_embs)
    print(f"Embeddings saved to {args.output_dir} (ts_level shape example: {ts_embs[infer_dataset.sequence_names[0]].shape}), instance_level shape example: {inst_embs[infer_dataset.sequence_names[0]].shape}), sliding shape example: {slide_embs[infer_dataset.sequence_names[0]].shape})")