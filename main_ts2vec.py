from ts2vec import TS2Vec
import numpy as np
from tqdm import tqdm
import argparse
import torch
import os


def get_args_parser():
    parser = argparse.ArgumentParser("TS2Vec pre-training", add_help=False)

    # Data
    parser.add_argument("--dataset", default="shot7m2", choices=["arena", "shot7m2"], type=str)
    parser.add_argument("--train_data_path", default="/scratch/izar/boesch/data/Shot7M2/train/train_dictionary_poses.npy", type=str)
    parser.add_argument("--inference_data_path", default="/scratch/izar/boesch/data/Shot7M2/test/test_dictionary_poses.npy", type=str)

    parser.add_argument("--output_dir", default="/scratch/izar/boesch/ts2vec/outputs/shot7m2", type=str)
    parser.add_argument("--num_frames", default=900, type=int, help="Number of frames per window")
    parser.add_argument("--split_tokenization", action="store_true", help="Use split tokenization (supported by SHOT7M2 dataset)")

    # Model
    parser.add_argument("--output_dims", default=128, type=int, help="Embedding dimension")
    parser.add_argument("--hidden_dims", default=64, type=int, help="Hidden dimension in encoder")
    parser.add_argument("--depth", default=10, type=int, help="Number of dilated conv layers")

    # Training
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--max_train_length", default=None, type=int, help="Max sequence length during training (for memory)")
    parser.add_argument("--n_train_samples", default=10000, type=int, help="Number of windows to subsample for training (set to -1 to use all)")
    parser.add_argument("--device", default=0, type=int, help="GPU device id")
    parser.add_argument("--num_workers", default=8, type=int)

    # Inference / sliding window
    parser.add_argument("--encoding_window", default=1 , type=int, help="kernel size used for max pooling during encoding (1 means no pooling, i.e. full temporal resolution)")
    parser.add_argument("--sliding_length", default=1, type=int, help="Sliding inference step size")
    parser.add_argument("--sliding_padding", default=50, type=int, help="Context padding for sliding inference")

    return parser


def _prepare_sequence_for_ts2vec(sequence):
    """Convert dataset-specific sequence tensors to TS2Vec shape (1, T, F)."""
    seq = np.asarray(sequence, dtype=np.float32)

    if seq.ndim == 2:
        return seq[None, ...]  # add batch dim

    # Arena inference format: (T, K, D) -> (1, T, K*D)
    if seq.ndim == 3:
        seq = seq.reshape(1, seq.shape[0], -1)
        return seq

    # SHOT7M2 inference format: (1, 1, T, 1, F) -> (1, T, F)
    if seq.ndim == 5 and seq.shape[0] == 1 and seq.shape[1] == 1 and seq.shape[3] == 1:
        seq = seq[0, 0, :, 0, :]
        return seq[None, ...]

    # Legacy flattened format occasionally seen in experiments: (1, T, 1, F) -> (1, T, F)
    if seq.ndim == 4 and seq.shape[0] == 1 and seq.shape[2] == 1:
        seq = seq[0, :, 0, :]
        return seq[None, ...]

    raise ValueError(f"Unsupported sequence shape for inference: {seq.shape}")


def _build_datasets(args):
    if args.dataset == "arena":
        from datasets.arena_dataset import ArenaDataset

        # Arena pretrain dataset enumerates all windows in memory, which can OOM on large runs.
        # We therefore build only the inference dataset here and sample training windows directly
        # from the source file in extract_train_data_arena.
        train_dataset = None
        infer_dataset = ArenaDataset(
            path_to_data_dir=args.inference_data_path,
            mode="inference",
        )
        input_dims = ArenaDataset.NUM_KEYPOINTS * ArenaDataset.KPTS_DIMENSIONS
    elif args.dataset == "shot7m2":
        from datasets.shot7m2 import SHOT7M2Dataset

        train_dataset = SHOT7M2Dataset(
            path_to_data_dir=args.train_data_path,
            mode="pretrain",
            num_frames=args.num_frames,
            split_tokenization=args.split_tokenization,
        )
        infer_dataset = SHOT7M2Dataset(
            path_to_data_dir=args.inference_data_path,
            mode="inference",
            split_tokenization=args.split_tokenization,
        )
        input_dims = train_dataset.NUM_KEYPOINTS * train_dataset.KPTS_DIMENSIONS
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_dataset, infer_dataset, input_dims


def extract_train_data_arena(path, num_frames, n_samples):
    """Randomly sample valid Arena windows directly from npz without building all windows in memory."""
    if n_samples <= 0:
        raise ValueError(
            "Arena training requires --n_train_samples > 0 to avoid loading all windows in memory."
        )

    with np.load(path, allow_pickle=True) as data:
        keypoints_dict = data["keypoints"].item()

    sequences = []
    num_possible_windows = []
    for seq in keypoints_dict.values():
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[0] < num_frames:
            continue
        flat = seq.reshape(seq.shape[0], -1)
        n_windows = flat.shape[0] - num_frames + 1
        if n_windows > 0:
            sequences.append(flat)
            num_possible_windows.append(n_windows)

    if not sequences:
        raise ValueError("No Arena sequences were long enough to sample training windows.")

    weights = np.asarray(num_possible_windows, dtype=np.float64)
    weights /= weights.sum()

    rng = np.random.default_rng(0)
    sampled = []
    max_attempts = max(n_samples * 20, 10000)

    for _ in tqdm(range(max_attempts), desc="Sampling Arena windows"):
        if len(sampled) >= n_samples:
            break

        seq_ix = rng.choice(len(sequences), p=weights)
        seq = sequences[seq_ix]
        start = rng.integers(0, seq.shape[0] - num_frames + 1)
        window = seq[start : start + num_frames]
        if not np.isnan(window).any():
            sampled.append(window)

    if not sampled:
        raise RuntimeError("Failed to sample any valid (non-NaN) Arena training windows.")

    if len(sampled) < n_samples:
        print(
            f"Warning: requested {n_samples} windows but sampled only {len(sampled)} valid windows "
            f"after {max_attempts} attempts."
        )

    train_data = np.stack(sampled, axis=0)
    print(f"Train data shape: {train_data.shape}  ({train_data.nbytes / 1e9:.2f} GB)")
    return train_data


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


def run_inference_for_mode(model, infer_dataset, args, mode):
    embeddings = {}

    for sequence_name, sequence in tqdm(
        zip(infer_dataset.sequence_names, infer_dataset.sequences),
        total=len(infer_dataset.sequences),
        desc=f"Inference ({mode})"
    ):
        seq = _prepare_sequence_for_ts2vec(sequence)

        if mode == "ts":
            emb = model.encode(seq)
        elif mode == "instance":
            emb = model.encode(seq, encoding_window="full_series")
        elif mode == "sliding":
            emb = model.encode(
                seq,
                causal=True,
                sliding_length=args.sliding_length,
                sliding_padding=args.sliding_padding,
            )
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        embeddings[sequence_name] = emb[0]
        del emb, seq

    return embeddings

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Datasets
    print(f"Loading datasets for: {args.dataset}")
    train_dataset, infer_dataset, input_dims = _build_datasets(args)
    print(f"Using input_dims={input_dims}")

    # Extract training data (subsampled if needed)
    if args.dataset == "arena":
        train_data = extract_train_data_arena(
            path=args.train_data_path,
            num_frames=args.num_frames,
            n_samples=args.n_train_samples,
        )
    else:
        train_data = extract_train_data(
            train_dataset,
            n_samples=args.n_train_samples,
            num_workers=args.num_workers,
            batch_size=args.batch_size * 8,  # larger batch for extraction only
        )

    # Model
    # check if model exists
    model_path = os.path.join(args.output_dir, "model.pkl")
    
    # Convert device int to torch.device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        # Create instance first, then load state dict
        model = TS2Vec(
            input_dims=input_dims,
            device=device,
            output_dims=args.output_dims,
            hidden_dims=args.hidden_dims,
            depth=args.depth,
            lr=args.lr,
            batch_size=args.batch_size,
            max_train_length=args.max_train_length,
        )
        model.load(model_path)
    else:
        model = TS2Vec(
            input_dims=input_dims,
            device=device,
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

    # Free training tensors before inference to reduce peak memory.
    del train_data

    # Inference
    print("Running inference (ts level)...")
    ts_embs = run_inference_for_mode(model, infer_dataset, args, mode="ts")
    np.save(os.path.join(args.output_dir, "ts_level_embeddings.npy"), ts_embs)
    ts_example_shape = ts_embs[infer_dataset.sequence_names[0]].shape
    del ts_embs

    print("Running inference (instance level)...")
    inst_embs = run_inference_for_mode(model, infer_dataset, args, mode="instance")
    np.save(os.path.join(args.output_dir, "instance_level_embeddings.npy"), inst_embs)
    inst_example_shape = inst_embs[infer_dataset.sequence_names[0]].shape
    del inst_embs

    print("Running inference (sliding)...")
    slide_embs = run_inference_for_mode(model, infer_dataset, args, mode="sliding")
    np.save(os.path.join(args.output_dir, "ts_level_sliding_embeddings.npy"), slide_embs)
    slide_example_shape = slide_embs[infer_dataset.sequence_names[0]].shape
    del slide_embs

    print(
        f"Embeddings saved to {args.output_dir} "
        f"(ts_level shape example: {ts_example_shape}), "
        f"instance_level shape example: {inst_example_shape}), "
        f"sliding shape example: {slide_example_shape})"
    )