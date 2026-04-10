import argparse
import datetime
import json
from math import prod
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from iopath.common.file_io import g_pathmgr as pathmgr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.hbabel import hBABELDataset
from datasets.mabe22_mice import MABeMouseDataset
from datasets.shot7m2 import SHOT7M2Dataset
from datasets.arena_dataset import ArenaDataset
from engine_pretrain import train_one_epoch
from models import models_defs
from util import misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import parse_tuples, str2bool



def get_args_parser():
    parser = argparse.ArgumentParser("sequence length agnostic hBehaveMAE embeddings extraction ", add_help=False)
    parser.add_argument(
        "--dataset",
        default="shot7m2",
        type=str,
        help="Type of dataset [mabe_mice, shot7m2, hbabel, arena]",
    )

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="gen_hiera",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument(
        "--path_to_data_dir",
        default="",
        help="path where to load data from",
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save",
    )
    parser.add_argument(
        "--log_dir",
        default="",
        help="path where to tensorboard log",
    )
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--sampling_rate", default=1, type=int)

    # hBehaveMAE specific parameters
    parser.add_argument("--input_size", default=(1, 54), nargs="+", type=int)
    parser.add_argument("--stages", default=(2, 3, 4), nargs="+", type=int)
    parser.add_argument(
        "--q_strides", default=[(1, 1, 3), (1, 1, 4), (1, 3, 1)], type=parse_tuples
    )
    parser.add_argument(
        "--mask_unit_attn", default=(True, False, False), nargs="+", type=str2bool
    )
    parser.add_argument("--patch_kernel", default=(4, 1, 2), nargs="+", type=int)
    parser.add_argument("--init_embed_dim", default=48, type=int)
    parser.add_argument("--init_num_heads", default=2, type=int)
    parser.add_argument("--out_embed_dims", default=(32, 64, 96), nargs="+", type=int)

    parser.add_argument("--fill_holes", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument(
        "--pos_only",
        action="store_true",
        help="Use only center position and orientation features for Arena centeralign mode.",
    )
    parser.add_argument(
        "--no_pos",
        action="store_true",
        help="Drop absolute position features and keep only non-position centeralign features.",
    )
    parser.add_argument(
        "--subsample_keypoints",
        action="store_true",
        help="For Arena, keep only a subset of keypoints (e.g. head and body) to reduce input dimensionality.",
    )
    parser.add_argument(
        "--window_size_embedding",
        default=1,
        type=int,
        help="Window size for averaging embeddings before saving (in frames). Larger values = more averaging = smaller final embeddings.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether to use bootstrapping of final embeddings by padding left and right with copies and stride 1 for the length of the mask unit."

    )
    parser.add_argument(
    "--no_batch",
    action="store_true",
    help="Process bootstrap offsets sequentially instead of batching them. Use when embeddings are too large to fit in memory.",
    )
    parser.add_argument(
        "--half_batch",
        action="store_true",
        help="When bootstrapping with batching, split the batch into two halves to reduce memory usage. Use when full batch still doesn't fit in memory.",
    )
    parser.add_argument(
            "--max_nan_frac",
            default=0.0,
            type=float,
            help="Maximum fraction of NaN values allowed in a sequence before it's discarded. Applied after featurization and before embedding extraction.",
        )

    parser.add_argument("--no_qkv_bias", action="store_true")
    return parser


def _window_and_average_embedding(emb, window_size):
    """Average embeddings within windows.
    
    Args:
        emb: numpy array with shape [num_frames, ...].
            Any spatial/token dims after the first are flattened into features.
        window_size: int, frames to average per window
    
    Returns:
        windowed_emb: numpy array of shape [num_windows, flattened_feature_dim]
        where num_windows = ceil(num_frames / window_size)
    """
    emb = np.asarray(emb)
    if emb.ndim < 2:
        raise ValueError(f"Expected embedding with at least 2 dims [T, ...], got {emb.shape}")
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    num_frames = emb.shape[0]
    emb_2d = emb.reshape(num_frames, -1)
    embedding_dim = emb_2d.shape[1]
    num_windows = (num_frames + window_size - 1) // window_size  # ceil division
    
    # Pad embeddings to multiple of window_size
    padded_size = num_windows * window_size
    if padded_size > num_frames:
        pad_width = ((0, padded_size - num_frames), (0, 0))
        # Pad with last frame value (nearest neighbor padding)
        emb_padded = np.pad(emb_2d, pad_width, mode='edge')
    else:
        emb_padded = emb_2d
    
    # Reshape to [num_windows, window_size, embedding_dim]
    emb_reshaped = emb_padded.reshape(num_windows, window_size, embedding_dim)
    
    # Average along window axis
    windowed_emb = emb_reshaped.mean(axis=1)  # [num_windows, embedding_dim]
    
    return windowed_emb

def load_model(args):

    # Device configurations
    device = torch.device(args.device)

    model = models_defs.__dict__[args.model](
        **vars(args),
    )
    # load last model checkpoint
    chkpt = misc.get_last_checkpoint(args)

    with pathmgr.open(chkpt, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu", weights_only=False)

    print("Load pre-trained checkpoint from: %s" % args.output_dir)
    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint["model_state"]

    # interpolate position embedding
    # interpolate_pos_embed(model, checkpoint_model) TODO : WHy ? 
    checkpoint_model = misc.convert_checkpoint(checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    model = model.eval()

    return model, device

def load_data(args):
    if args.dataset == "mabe_mice":
        pass
    elif args.dataset == "shot7m2":
        dataset = SHOT7M2Dataset(path_to_data_dir=args.path_to_data_dir, mode="inference", sampling_rate=args.sampling_rate, split_tokenization=True)
        pass
    elif args.dataset == "hbabel":
        pass
    elif args.dataset == "arena":
        dataset = ArenaDataset(
            path_to_data_dir=args.path_to_data_dir,
            mode="inference",
            sampling_rate=args.sampling_rate,
            centeralign=args.centeralign,
            pos_only=args.pos_only,
            no_pos=args.no_pos,
            subsample_keypoints=args.subsample_keypoints,
            max_nan_frac=args.max_nan_frac
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return dataset


def _prepare_sequence_for_model(sequence: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Convert sequence to expected 5D shape [B, C, T, N, D] for Conv3d patch embedding."""
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.tensor(sequence, dtype=torch.float32)
    else:
        sequence = sequence.float()

    # Arena inference often yields flattened [T, F].
    # Map to [B=1, C=1, T, N=1, D=F].
    if sequence.ndim == 2:
        sequence = sequence.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        return sequence

    # Sometimes data may come as [T, K, D].
    # Flatten spatial dims -> [T, F], then map to [1, 1, T, 1, F].
    if sequence.ndim == 3:
        if dataset_name == "arena":
            t = sequence.shape[0]
            sequence = sequence.reshape(t, -1).unsqueeze(0).unsqueeze(0).unsqueeze(3)
        else:
            # Conservative fallback: treat first dim as batch.
            sequence = sequence.unsqueeze(1).unsqueeze(3)
        return sequence

    # Legacy format [B, T, N, D] -> [B, C=1, T, N, D]
    if sequence.ndim == 4:
        sequence = sequence.unsqueeze(1)
        return sequence

    # Expected format already: [B, C, T, N, D]
    if sequence.ndim == 5:
        return sequence

    raise ValueError(
        f"Unsupported input shape {tuple(sequence.shape)} for dataset={dataset_name}. "
        "Expected a tensor convertible to [B, C, T, N, D]."
    )

def _bootstrap_and_extract(model, sequence, device, pad_size, stages, no_batch=False, half_batch=False):
    print(f"Applying bootstrapping with pad_size={pad_size}")

    cumulative_strides = [
        model.patch_stride[0] * prod(model.q_strides[n_i][0] for n_i in range(n))
        for n in range(len(stages))
    ]

    if no_batch:
        accumulated = None
        for i in range(pad_size):
            padded = torch.nn.functional.pad(
                sequence, (0, 0, 0, 0, i, pad_size - i), mode='replicate'
            )
            _, embs_padded = model(padded.to(device), return_intermediates=True, inference=True)

            crops = []
            for n, layer_emb in enumerate(embs_padded):
                stride = cumulative_strides[n]
                t_pad_tokens = pad_size // stride
                t_original = layer_emb.shape[1] - t_pad_tokens
                t_offset = i // stride
                crops.append(layer_emb[:, t_offset: t_offset + t_original].detach())  # ← detach

            if accumulated is None:
                accumulated = [c.clone() for c in crops]
            else:
                for n in range(len(crops)):
                    accumulated[n] += crops[n]

            del padded, embs_padded, crops

        hierarchical_embeddings = [acc / pad_size for acc in accumulated]

    else:
        # Original batched path
        padded_sequences = []
        for i in range(pad_size):
            padded_sequences.append(
                torch.nn.functional.pad(sequence, (0, 0, 0, 0, i, pad_size - i), mode='replicate')
            )
        batched_sequence = torch.cat(padded_sequences, dim=0)
        del padded_sequences  # free the list of padded sequences immediately

        if half_batch:
            quarter = pad_size // 4
            quarters = [
                batched_sequence[quarter*i : quarter*(i+1)]
                for i in range(4)
            ]
            # free the full batch immediately
            del batched_sequence
            torch.cuda.empty_cache()

            accumulated_embs = None
            for q in quarters:
                print(f"Processing quarter with shape {tuple(q.shape)}")
                _, embs = model(q.to(device), return_intermediates=True, inference=True)
                
                embs = [e.cpu() for e in embs]  # move to CPU immediately, pas besoin de detach() avec inference_mode()
                if accumulated_embs is None:
                    accumulated_embs = embs
                else:
                    for n in range(len(embs)):
                        accumulated_embs[n] = torch.cat([accumulated_embs[n], embs[n]], dim=0)
                del q, embs
                torch.cuda.empty_cache()

            hierarchical_embeddings_padded = accumulated_embs
        else:
            _, hierarchical_embeddings_padded = model(
                batched_sequence.to(device), return_intermediates=True, inference=True
            )

        hierarchical_embeddings = []
        for n, layer_emb in enumerate(hierarchical_embeddings_padded):
            stride = cumulative_strides[n]
            t_pad_tokens = pad_size // stride
            t_original = layer_emb.shape[1] - t_pad_tokens
            crops = []
            for i in range(pad_size):
                t_offset = i // stride
                crops.append(layer_emb[i:i+1, t_offset: t_offset + t_original])
            hierarchical_embeddings.append(torch.mean(torch.cat(crops, dim=0), dim=0, keepdim=True))
    
    return hierarchical_embeddings

def extract_hierarchical_embeddings(args):

    model, device = load_model(args)
    dataset = load_data(args)

    layer_embeddings = {}
    window_size = int(args.window_size_embedding)

    token_shapes = [model.get_layer_token_shape(i) for i in range(len(args.stages))]

    for seq_idx, (sequence_name, sequence) in enumerate(
        tqdm(zip(dataset.sequence_names, dataset.sequences), total=len(dataset.sequences)),
        start=1,
    ):
        sequence = dataset.featurise_keypoints(sequence)
        sequence = _prepare_sequence_for_model(sequence, args.dataset)
        print(f"Processing sequence {seq_idx}/{len(dataset.sequences)}: '{sequence_name}' with shape {tuple(sequence.shape)}")
        # move to device

        with torch.inference_mode():

            if args.bootstrap:
                pad_size = model.temporal_pooling_factor * model.patch_stride[0]
                hierarchical_embeddings = _bootstrap_and_extract(model, sequence, device, pad_size, args.stages, no_batch=args.no_batch, half_batch=args.half_batch)

            else:
                _, hierarchical_embeddings = model(sequence.to(device), return_intermediates=True, inference=True)

        # Accumulate per layer (en dehors du bloc inference_mode, sur CPU)
        for i, emb in enumerate(hierarchical_embeddings):
            # Remove batch dimension
            emb = emb.squeeze(0)  # [1, ...] -> [...]
            
            # Squeeze all singleton spatial/mask-unit dimensions, keep last (features)
            while emb.ndim > 2:
                if emb.shape[0] == 1:
                    emb = emb.squeeze(0)
                elif emb.shape[1] == 1:
                    emb = emb.squeeze(1)
                else:
                    break
             
            
            # Apply windowing and averaging
            w_s = max(1, window_size // token_shapes[i][0])  # Adjust window size based on token shape
            emb_np = emb.cpu().numpy() # Transfert propre vers numpy
            if w_s > 1:
                print(f"Applying windowing with window_size={window_size} (adjusted to {w_s} for layer {i} with token shape {token_shapes[i]})")
                emb_np = _window_and_average_embedding(emb_np, w_s)
            else:
                # Keep full temporal resolution, but normalize to [T, F] even when spatial dims remain.
                emb_np = emb_np.reshape(emb_np.shape[0], -1)
            
            layer_key = f"layer_{i}"
            if layer_key not in layer_embeddings:
                layer_embeddings[layer_key] = {}
            layer_embeddings[layer_key][sequence_name] = emb_np

        # Explicitly release GPU tensors each iteration to keep VRAM stable.
        del sequence, hierarchical_embeddings
        if args.bootstrap and not args.no_batch:
            if 'batched_sequence' in locals(): del batched_sequence
            if 'hierarchical_embeddings_padded' in locals(): del hierarchical_embeddings_padded
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Adjust token_shapes to reflect windowed output
    token_shapes = [(token_shapes[i][0]*window_size, 1, 1) for i in range(len(args.stages))]
    np.save(os.path.join(args.output_dir, "token_shapes.npy"), token_shapes)
    embed_path = os.path.join(args.output_dir, f"embeddings{'_no_bootstrap' if not args.bootstrap else ''}.npy")
    np.save(embed_path, layer_embeddings)
    print(f"Saved windowed hierarchical embeddings to {args.output_dir} ({len(layer_embeddings)} layers, window_size={window_size})")
