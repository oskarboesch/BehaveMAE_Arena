import os
import numpy as np
import pandas as pd


def _normalize_per_layer_param(value, num_layers, name):
    """Expand an int or validate a per-layer list/tuple for all layers."""
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
        return [value] * num_layers

    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != num_layers:
            raise ValueError(
                f"{name} must have one value per layer ({num_layers}), got {len(value)}."
            )
        values = [int(v) for v in value]
        if any(v < 1 for v in values):
            raise ValueError(f"All {name} values must be >= 1.")
        return values

    raise TypeError(f"{name} must be an int or a list/tuple/array of ints.")


def _chunk_average(run_embeddings, chunk_size, stride):
    """Average embeddings over sliding chunks for one run."""
    num_frames = run_embeddings.shape[0]
    if num_frames == 0:
        return np.empty((0, run_embeddings.shape[1]), dtype=run_embeddings.dtype)

    if num_frames < chunk_size:
        return run_embeddings.mean(axis=0, keepdims=True)

    starts = np.arange(0, num_frames - chunk_size + 1, stride)
    chunk_means = [
        run_embeddings[start : start + chunk_size].mean(axis=0) for start in starts
    ]
    return np.stack(chunk_means, axis=0)


def _load_kinematics_dict(kinematics_path):
    """Load kinematics dictionary from .npz/.npy files."""
    if not os.path.exists(kinematics_path):
        raise FileNotFoundError(f"Kinematics file not found: {kinematics_path}")

    if kinematics_path.endswith(".npz"):
        data = np.load(kinematics_path, allow_pickle=True)
        if "kinematics" in data:
            return data["kinematics"].item()
        if len(data.files) == 1:
            return data[data.files[0]].item()
        raise KeyError(
            f"Could not find 'kinematics' key in {kinematics_path}. Available keys: {data.files}"
        )

    if kinematics_path.endswith(".npy"):
        data = np.load(kinematics_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
            return data.item()
        if isinstance(data, dict):
            return data
        raise TypeError(
            f"Unsupported .npy kinematics structure in {kinematics_path}. Expected dict-like object."
        )

    raise ValueError(
        f"Unsupported kinematics file extension for {kinematics_path}. Use .npz or .npy"
    )


def _run_kinematics_to_matrix(run_kinematics, target_len=None):
    """Convert one run kinematics dict to (time, features) matrix.

    If target_len is provided, each feature is left-padded with zeros so all
    kinematics time series match the embeddings temporal length for that run.
    """
    if not isinstance(run_kinematics, dict) or len(run_kinematics) == 0:
        if target_len is None:
            return np.empty((0, 0), dtype=np.float32), []
        return np.zeros((target_len, 0), dtype=np.float32), []

    feature_names = list(run_kinematics.keys())
    feature_arrays = [np.asarray(run_kinematics[name]).reshape(-1) for name in feature_names]

    if target_len is None:
        min_len = min(arr.shape[0] for arr in feature_arrays)
        if min_len == 0:
            return np.empty((0, len(feature_names)), dtype=np.float32), feature_names
        stacked = np.stack([arr[:min_len] for arr in feature_arrays], axis=1)
        return stacked, feature_names

    padded_features = []
    for arr in feature_arrays:
        arr = arr.astype(np.float32, copy=False)
        if arr.shape[0] < target_len:
            pad_len = target_len - arr.shape[0]
            arr = np.pad(arr, (pad_len, 0), mode="constant", constant_values=0)
        elif arr.shape[0] > target_len:
            arr = arr[:target_len]
        padded_features.append(arr)

    stacked = np.stack(padded_features, axis=1)
    return stacked, feature_names

def load_numpy_embeddings(path, num_runs=None, chunk_size=1, stride=1, meta_data_path=None, kinematics_path=None):
    """
    Load embeddings from .npy files and return a list of embeddings and metadata dataframe.
    
    Args:
        path (str): Path to the directory containing the .npy files.
        num_runs (int, optional): Number of runs to load. If None, load all runs. Default is None.
        chunk_size (int, optional): Number of frames in each chunk. Default is 1.
        stride (int, optional): Stride between chunks. Default is 1.
        meta_data_path (str, optional): Path to the metadata CSV file. If None, metadata will not be loaded. Default is None.

    Returns:
        list_of_embeddings (list): List of numpy arrays containing the embeddings for each run.
        metadata_df (pd.DataFrame or None): DataFrame containing the metadata and frame_number map additional metadata if meta_data_path is provided.
        chunked_kinematics (np.ndarray or None): Chunk-averaged kinematics aligned by selected runs if kinematics_path is provided.
    """
    list_of_embeddings = []

    # List all test files in the directory, there is one for each embedding layer.
    npy_files = [
        f for f in os.listdir(path) if f.startswith("embeddings_layer_") and f.endswith(".npy")
    ]
    if len(npy_files) == 0:
        raise FileNotFoundError(f"No embeddings_layer_*.npy files found in: {path}")

    def _layer_sort_key(filename):
        stem = os.path.splitext(filename)[0]
        suffix = stem.split("embeddings_layer_")[-1]
        digits = "".join(ch for ch in suffix if ch.isdigit())
        return (0, int(digits)) if digits else (1, stem)

    npy_files = sorted(npy_files, key=_layer_sort_key)

    # Load all layers first.
    layer_data = []
    for npy_file in npy_files:
        file_path = os.path.join(path, npy_file)
        data = np.load(file_path, allow_pickle=True).item()
        if "embeddings" not in data or "frame_number_map" not in data:
            raise KeyError(
                f"File {file_path} must contain keys 'embeddings' and 'frame_number_map'."
            )
        layer_data.append(data)

    num_layers = len(layer_data)
    chunk_sizes = _normalize_per_layer_param(chunk_size, num_layers, "chunk_size")
    strides = _normalize_per_layer_param(stride, num_layers, "stride")

    # Use layer 0 as canonical run ordering.
    canonical_map = layer_data[0]["frame_number_map"]
    selected_run_ids = list(canonical_map.keys())
    if num_runs is not None:
        selected_run_ids = selected_run_ids[:num_runs]

    # Process each layer run-by-run with layer-specific chunk/stride.
    per_layer_new_ranges = [dict() for _ in range(num_layers)]
    for layer_idx, data in enumerate(layer_data):
        frame_number_map = data["frame_number_map"]
        embeddings = data["embeddings"]
        layer_chunks = []
        cursor = 0

        for run_id in selected_run_ids:
            if run_id not in frame_number_map:
                raise KeyError(
                    f"Run '{run_id}' missing from layer {layer_idx} frame_number_map."
                )

            start_idx, end_idx = frame_number_map[run_id]
            run_embeddings = embeddings[start_idx:end_idx]
            run_chunked = _chunk_average(
                run_embeddings,
                chunk_size=chunk_sizes[layer_idx],
                stride=strides[layer_idx],
            )
            layer_chunks.append(run_chunked)

            per_layer_new_ranges[layer_idx][run_id] = (cursor, cursor + len(run_chunked))
            cursor += len(run_chunked)

        if len(layer_chunks) == 0:
            list_of_embeddings.append(np.empty((0, embeddings.shape[1]), dtype=embeddings.dtype))
        else:
            list_of_embeddings.append(np.concatenate(layer_chunks, axis=0))

    # Build metadata table for selected runs.
    metadata_rows = []
    for run_id in selected_run_ids:
        parts = run_id.split("_")
        animal_id = parts[0] if len(parts) > 0 else run_id
        experimental_phase = parts[1] if len(parts) > 1 else "unknown"
        age_phase = parts[2] if len(parts) > 2 else "unknown"
        mouse_run_id = parts[3] if len(parts) > 3 else "unknown"
        original_range = canonical_map[run_id]

        row = {
            "run_id": run_id,
            "animal_id": animal_id,
            "experimental_phase": experimental_phase,
            "age_phase": age_phase,
            "mouse_run_id": mouse_run_id,
            "frame_range": original_range,
            "new_frame_range": per_layer_new_ranges[0][run_id],
            "strain": None,
            "strain_family": None,
        }

        # Keep per-layer remapped ranges explicit when chunking differs by layer.
        for layer_idx in range(num_layers):
            row[f"new_frame_range_layer{layer_idx}"] = per_layer_new_ranges[layer_idx][run_id]

        metadata_rows.append(row)

    metadata_df = pd.DataFrame(metadata_rows)

    chunked_kinematics = None
    kinematics_feature_names = None
    kinematics_new_ranges = {}
    if kinematics_path is not None:
        kinematics_dict = _load_kinematics_dict(kinematics_path)
        kinematics_chunks = []
        cursor = 0

        for run_id in selected_run_ids:
            if run_id not in kinematics_dict:
                raise KeyError(f"Run '{run_id}' missing from kinematics data.")

            run_start, run_end = canonical_map[run_id]
            target_len = int(run_end - run_start)
            run_matrix, run_feature_names = _run_kinematics_to_matrix(
                kinematics_dict[run_id],
                target_len=target_len,
            )
            if kinematics_feature_names is None:
                kinematics_feature_names = run_feature_names
            elif run_feature_names != kinematics_feature_names:
                raise ValueError(
                    f"Kinematics feature mismatch for run '{run_id}'. "
                    f"Expected {kinematics_feature_names}, got {run_feature_names}."
                )

            run_chunked_kinematics = _chunk_average(
                run_matrix,
                chunk_size=chunk_sizes[0],
                stride=strides[0],
            )

            expected_start, expected_end = per_layer_new_ranges[0][run_id]
            expected_chunks = expected_end - expected_start
            if len(run_chunked_kinematics) != expected_chunks:
                raise ValueError(
                    f"Chunked kinematics length mismatch for run '{run_id}': "
                    f"got {len(run_chunked_kinematics)}, expected {expected_chunks}."
                )

            kinematics_chunks.append(run_chunked_kinematics)

            kinematics_new_ranges[run_id] = (cursor, cursor + len(run_chunked_kinematics))
            cursor += len(run_chunked_kinematics)

        if len(kinematics_chunks) == 0:
            num_features = 0 if kinematics_feature_names is None else len(kinematics_feature_names)
            chunked_kinematics = np.empty((0, num_features), dtype=np.float32)
        else:
            chunked_kinematics = np.concatenate(kinematics_chunks, axis=0)

        if len(metadata_df) > 0:
            metadata_df["new_frame_range_kinematics"] = metadata_df["run_id"].map(kinematics_new_ranges)

    if meta_data_path is not None and len(metadata_df) > 0:
        mdata_df = pd.read_csv(meta_data_path, sep="\t")
        if "animal_id" in mdata_df.columns and "strain" in mdata_df.columns:
            strain_map = mdata_df.drop_duplicates("animal_id").set_index("animal_id")["strain"]
            metadata_df["strain"] = metadata_df["animal_id"].map(strain_map).fillna("unknown")
        else:
            metadata_df["strain"] = "unknown"

        if "animal_id" in mdata_df.columns and "strain_family" in mdata_df.columns:
            family_map = mdata_df.drop_duplicates("animal_id").set_index("animal_id")["strain_family"]
            metadata_df["strain_family"] = metadata_df["animal_id"].map(family_map).fillna("unknown")
        else:
            metadata_df["strain_family"] = "unknown"

    # Descriptive summary of what was loaded and produced.
    print("\n=== Loaded Embeddings Summary ===")
    print(f"Source directory: {path}")
    print(f"Layer files: {npy_files}")
    print(f"Selected runs: {len(selected_run_ids)}")
    if len(selected_run_ids) > 0:
        preview = selected_run_ids[:3]
        suffix = " ..." if len(selected_run_ids) > 3 else ""
        print(f"Run preview: {preview}{suffix}")

    print(f"Chunk size per layer: {chunk_sizes}")
    print(f"Stride per layer: {strides}")
    for layer_idx, emb in enumerate(list_of_embeddings):
        print(
            f"Layer {layer_idx}: shape={emb.shape}, "
            f"dtype={emb.dtype}, runs={len(per_layer_new_ranges[layer_idx])}"
        )

    print(f"Metadata rows: {len(metadata_df)}")
    if len(metadata_df) > 0:
        print(f"Metadata columns: {list(metadata_df.columns)}")

    if chunked_kinematics is not None:
        print(
            "Kinematics: "
            f"shape={chunked_kinematics.shape}, "
            f"features={kinematics_feature_names}, "
            f"chunk_size={chunk_sizes[0]}, stride={strides[0]}"
        )

    return list_of_embeddings, metadata_df, chunked_kinematics