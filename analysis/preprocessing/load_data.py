import os
import numpy as np
import pandas as pd
from datasets.keypoints import get_kinematics
from datasets.syllables import load_kpt_moseq
from datasets.arena_dataset import extract_metadata_from_runid


def _ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _layer_sort_key(filename):
    return int(filename.replace("test_submission_", "").replace(".npy", ""))


def _load_windowed_embeddings(path_to_emb_dir):
    embedding_files = sorted(
        [f for f in os.listdir(path_to_emb_dir) if f.startswith("test_submission_") and f.endswith(".npy")],
        key=_layer_sort_key,
    )
    if len(embedding_files) == 0:
        raise FileNotFoundError(
            f"No embedding files found in {path_to_emb_dir} with pattern 'test_submission_*.npy'."
        )

    embeddings = {f"layer_{i}": {} for i in range(len(embedding_files))}
    token_shapes = []

    for i, embedding_file in enumerate(embedding_files):
        layer_key = f"layer_{i}"
        embedding_path = os.path.join(path_to_emb_dir, embedding_file)
        data = np.load(embedding_path, allow_pickle=True)
        if "embeddings" not in data or "frame_number_map" not in data:
            raise KeyError(
                f"File {embedding_path} must contain keys 'embeddings' and 'frame_number_map'."
            )

        embed = data["embeddings"]
        frame_number_map = data["frame_number_map"]
        for run_id, frame_numbers in frame_number_map.items():
            frame_idx = np.arange(frame_numbers[0], frame_numbers[1])
            embeddings[layer_key][run_id] = embed[frame_idx]

        # Already frame-level embeddings for this format.
        token_shapes.append((1, 1, 1))

    return embeddings, token_shapes


def _load_full_sequence_embeddings(path_to_emb_dir):
    embedding_file = os.path.join(path_to_emb_dir, "embeddings.npy")
    token_shapes_file = os.path.join(path_to_emb_dir, "token_shapes.npy")
    _ensure_exists(token_shapes_file, "Token shapes file")

    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file, allow_pickle=True).item()
    else:
        shard_files = sorted(
            [
                f
                for f in os.listdir(path_to_emb_dir)
                if f.startswith("embeddings_shard_") and f.endswith(".npy")
            ]
        )
        if len(shard_files) == 0:
            raise FileNotFoundError(
                f"Embeddings file not found: {embedding_file}, and no shard files matching "
                f"'embeddings_shard_*.npy' found in {path_to_emb_dir}."
            )

        embeddings = {}
        for shard_file in shard_files:
            shard_path = os.path.join(path_to_emb_dir, shard_file)
            shard_data = np.load(shard_path, allow_pickle=True).item()
            for layer_key, layer_runs in shard_data.items():
                if layer_key not in embeddings:
                    embeddings[layer_key] = {}
                overlap = set(embeddings[layer_key].keys()).intersection(layer_runs.keys())
                if overlap:
                    raise ValueError(
                        f"Duplicate run ids found while merging shard {shard_file} for {layer_key}: "
                        f"{sorted(list(overlap))[:5]}"
                    )
                embeddings[layer_key].update(layer_runs)

    token_shapes = np.load(token_shapes_file, allow_pickle=True)
    return embeddings, token_shapes


def _load_ts2vec_dict(path_to_emb_dir, base_name):
    single_file = os.path.join(path_to_emb_dir, f"{base_name}.npy")
    if os.path.exists(single_file):
        return np.load(single_file, allow_pickle=True).item()

    shard_files = sorted(
        [
            f for f in os.listdir(path_to_emb_dir)
            if f.startswith(f"{base_name}_shard_") and f.endswith(".npy")
        ]
    )
    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"TS2Vec file not found: {single_file} and no shard files matching "
            f"'{base_name}_shard_*.npy' were found in {path_to_emb_dir}."
        )

    merged = {}
    for shard_file in shard_files:
        shard_path = os.path.join(path_to_emb_dir, shard_file)
        shard_data = np.load(shard_path, allow_pickle=True).item()
        overlap = set(merged.keys()).intersection(shard_data.keys())
        if overlap:
            raise ValueError(
                f"Duplicate run ids across shards for {base_name}: {sorted(list(overlap))[:5]}"
            )
        merged.update(shard_data)

    return merged


def _load_ts2vec_embeddings(path_to_emb_dir):
    source_names = [
        "ts_level_embeddings",
        "ts_level_sliding_embeddings",
        "instance_level_embeddings",
    ]
    embeddings = {}

    for i, source_name in enumerate(source_names):
        layer_key = f"layer_{i}"
        data = _load_ts2vec_dict(path_to_emb_dir, source_name)
        embeddings[layer_key] = {
            run_id: emb[np.newaxis, :] if emb.ndim == 1 else emb
            for run_id, emb in data.items()
        }

    # Frame-level except instance-level (single embedding vector per sequence).
    token_shapes = [(1, 1, 1), (1, 1, 1), (-1, 1, 1)]
    return embeddings, token_shapes


def _load_embeddings(path_to_emb_dir, embed_type):
    if embed_type == "windowed":
        return _load_windowed_embeddings(path_to_emb_dir)
    if embed_type == "full_sequence":
        return _load_full_sequence_embeddings(path_to_emb_dir)
    if embed_type == "ts2vec":
        return _load_ts2vec_embeddings(path_to_emb_dir)
    if embed_type == "no_bootstrap":
        return _load_full_sequence_embeddings(path_to_emb_dir.replace(".npy", "_no_bootstrap.npy"))
    raise ValueError(
        f"Unsupported embed_type: {embed_type}. Supported types are 'windowed', 'full_sequence', and 'ts2vec'."
    )


def _load_arena_metadata(run_ids, meta_data_path):
    metadata = pd.DataFrame([extract_metadata_from_runid(run_id) for run_id in run_ids])
    if meta_data_path is None:
        return metadata
    _ensure_exists(meta_data_path, "Metadata file")
    strain_metadata = pd.read_csv(meta_data_path, sep="\t")
    
    cols_to_merge = ["animal_id"]
    
    if "strain" in strain_metadata.columns:
        cols_to_merge.append("strain")
    
    if "family" in strain_metadata.columns:
        cols_to_merge.append("family")

    if "anxiety_level" in strain_metadata.columns:
        cols_to_merge.append("anxiety_level")

    if len(cols_to_merge) > 1 and "animal_id" in strain_metadata.columns:
        metadata = metadata.merge(
            strain_metadata[cols_to_merge],
            on="animal_id",
            how="left",
        )
    
    return metadata


def _load_kinematics(dataset, keypoints_path):
    if keypoints_path is None:
        return None

    _ensure_exists(keypoints_path, "Keypoints file")
    if dataset == "arena":
        keypoints = np.load(keypoints_path, allow_pickle=True)["keypoints"].item()
        kinematics = get_kinematics(keypoints, mean=False)
        return kinematics, keypoints

    if dataset == "shot7m2":
        keypoints = np.load(keypoints_path, allow_pickle=True).item()["sequences"]["keypoints"]
        keypoints = {run: kpt.squeeze(1) if kpt.ndim == 4 else kpt for run, kpt in keypoints.items()}
        kinematics = get_kinematics(keypoints, mean=False)
        return {
            run: {
                kinematic: values
                for kinematic, values in run_kinematics.items()
                if kinematic in ["speed_x", "speed_y", "speed", "acceleration"]
            }
            for run, run_kinematics in kinematics.items()
        }, keypoints

    raise ValueError(f"Unsupported dataset for keypoints/kinematics: {dataset}")


def _load_syllable_labels(path):
    if path is None:
        return None
    _ensure_exists(path, "Syllable labels file")
    return load_kpt_moseq(path, one_hot_encode=False)


def _load_shot7m2_true_labels():
    labels_path = "/scratch/izar/boesch/data/Shot7M2/test/benchmark_labels.npy"
    frame_number_map_path = "/scratch/izar/boesch/BehaveMAE/outputs/shot7m2/experiment1/test_submission_0.npy"

    if not os.path.exists(labels_path) or not os.path.exists(frame_number_map_path):
        print(
            "Warning: SHOT7M2 benchmark labels/frame map not found. "
            "Proceeding without true labels."
        )
        return None, None

    frame_number_map = np.load(frame_number_map_path, allow_pickle=True).item()["frame_number_map"]
    label_data = np.load(labels_path, allow_pickle=True).item()
    labels = label_data["label_array"]
    true_label_names = label_data["vocabulary"]
    true_labels = {
        key: labels[:, np.arange(values[0], values[1])].T
        for key, values in frame_number_map.items()
    }
    return true_labels, true_label_names


def _sample_runs_if_requested(args, embeddings, sequence_names, true_labels, metadata, kinematics, syllable_labels):
    if not args.sample:
        return embeddings, sequence_names, true_labels, metadata, kinematics, syllable_labels

    selected_runs = sequence_names[:2]
    for layer in embeddings.keys():
        embeddings[layer] = {
            run: embeddings[layer][run]
            for run in selected_runs
            if run in embeddings[layer]
        }

    if true_labels is not None:
        true_labels = {run: true_labels[run] for run in selected_runs if run in true_labels}

    if metadata is not None and "run_id" in metadata.columns:
        metadata = metadata[metadata["run_id"].isin(selected_runs)].copy()

    if kinematics is not None:
        kinematics = {run: kinematics[run] for run in selected_runs if run in kinematics}

    if syllable_labels is not None:
        syllable_labels = {run: syllable_labels[run] for run in selected_runs if run in syllable_labels}

    print(
        f"Loaded embeddings for {len(embeddings)} layers, {len(selected_runs)} sequences per layer "
        f"(selected runs: {selected_runs})"
    )
    return embeddings, selected_runs, true_labels, metadata, kinematics, syllable_labels


def load_data(args):
    if args.dataset not in ["arena", "shot7m2"]:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are 'arena' and 'shot7m2'.")

    embeddings, token_shapes = _load_embeddings(args.path_to_emb_dir, args.embed_type)
    sequence_names = list(embeddings[next(iter(embeddings))].keys())
    print(f"Loaded embeddings for {len(embeddings)} layers, {len(sequence_names)} runs/sequences per layer")

    true_label_names = None
    metadata = None
    true_labels = None
    keypoints = None
    kinematics = None

    if args.dataset == "arena":
        metadata = _load_arena_metadata(sequence_names, args.meta_data_path)
        print(f"Extracted metadata: {metadata.columns.tolist()}")
        true_labels = None

    elif args.dataset == "shot7m2":
        true_labels, true_label_names = _load_shot7m2_true_labels()

    if args.keypoints_path is not None:
        kinematics, keypoints = _load_kinematics(args.dataset, args.keypoints_path)
    if kinematics is not None and len(kinematics) > 0:
        first_key = next(iter(kinematics.keys()))
        print(f"Extracted kinematics: {kinematics[first_key].keys()}")

    syllable_labels = _load_syllable_labels(args.syllable_labels_path)

    embeddings, sequence_names, true_labels, metadata, kinematics, syllable_labels = _sample_runs_if_requested(
        args,
        embeddings,
        sequence_names,
        true_labels,
        metadata,
        kinematics,
        syllable_labels,
    )

    return embeddings, token_shapes, metadata, keypoints, kinematics, syllable_labels, true_labels, true_label_names