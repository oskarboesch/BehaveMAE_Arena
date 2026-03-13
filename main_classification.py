import argparse
import os
import random
from datetime import datetime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
# import downsampler
from util.logging import master_print as print

from imblearn.under_sampling import RandomUnderSampler

def get_args_parser():
    parser = argparse.ArgumentParser("Classification of the Arena Dataset", add_help=False)
    parser.add_argument(
        "--train_dataset_path",
        default="/scratch/izar/boesch/data/Arena_Data/kinematics-train.npz",
        type=str,
        help="Path to the training dataset",
    )
    parser.add_argument(
        "--test_dataset_path",
        default=None,
        type=str,
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--metadata_path",
        default="/scratch/izar/boesch/data/Arena_Data/hdp_meta.tsv",
        type=str,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--dataset",
        default="kinematics",
        type=str,
        help="Dataset to use: 'kinematics', 'embeddings', 'keypoints', or 'kp-moseq'",
    )
    parser.add_argument(
        "--classifier_model_type",
        default="logistic_regression",
        type=str,
        help="Type of classifier model to use",
    )
    parser.add_argument(
        "--window_size",
        default=500,
        type=int,
        help="Window size for sample generation",
    )
    parser.add_argument(
        "--window_stride",
        default=500,
        type=int,
        help="Stride for sliding windows",
    )
    parser.add_argument(
        "--n_windows_per_prediction",
        default=1,
        type=int,
        help="Number of windows used for one prediction",
    )
    parser.add_argument(
        "--agg_method",
        default="mean",
        type=str,
        help="Aggregation method: 'mean', 'median', or 'flatten'",
    )
    parser.add_argument(
        "--embedding_stage",
        default=None,
        type=str,
        help="Single embedding stage to use when --dataset embeddings (e.g. 'stage1')",
    )
    parser.add_argument(
        "--output_dir",
        default="/scratch/izar/boesch/classification_results",
        type=str,
        help="Directory to save classification results",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Global random seed for reproducible runs",
    )
    return parser


def set_reproducibility(seed):
    # Keep all local randomness controlled by one seed.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _extract_run_labels(run_id):
    parts = str(run_id).split("_")
    animal_id = parts[0] if len(parts) > 0 else "unknown_animal"
    exp = parts[1] if len(parts) > 1 else "unknown_exp"
    age = parts[2] if len(parts) > 2 else "unknown_age"
    return animal_id, exp, age


def _load_embeddings_h5(path, embedding_stage=None):
    """
    Loads embeddings from an H5 file with structure:
      /<run_id>/<stage_name> -> ndarray
    Example stage names: stage1, stage2, stage3
    """
    embeddings = {}
    with h5py.File(path, "r") as f:
        for run_id in tqdm(sorted(f.keys())):
            grp = f[run_id]
            if not isinstance(grp, h5py.Group):
                continue

            run_embeddings = {}
            # If embedding_stage is specified, only load that stage. Otherwise, load all stages.
            stages_to_load = [embedding_stage] if embedding_stage else sorted(grp.keys())
            for stage in stages_to_load:
                if stage in grp:
                    run_embeddings[stage] = np.asarray(grp[stage], dtype=np.float32)

            if run_embeddings:
                embeddings[run_id] = run_embeddings

    return embeddings


def _load_kpt_moseq(path):
    """
    Loads kpt-moseq data from an H5 file with structure:
      /<animal_id>/syllable -> ndarray
    Keeps only the 50 most common syllables and one-hot encodes them.
    Syllables not in top 50 are mapped to all zeros.
    """
    # First pass: find all syllables and count frequencies
    syllable_counts = {}
    with h5py.File(path, "r") as f:
        for animal_id in tqdm(sorted(f.keys()), desc="Counting syllables"):
            grp = f[animal_id]
            if "syllable" not in grp:
                continue
            syllables = np.asarray(grp["syllable"], dtype=np.int64)
            if syllables.ndim > 1:
                syllables = syllables.flatten()
            for syl in syllables:
                syl_id = int(syl)
                syllable_counts[syl_id] = syllable_counts.get(syl_id, 0) + 1

    # Get top 50 most common syllables
    top_50_syllables = sorted(syllable_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_syl_ids = [syl_id for syl_id, count in top_50_syllables]
    syl_to_idx = {syl_id: idx for idx, syl_id in enumerate(top_50_syl_ids)}
    
    print(f"Using top 50 syllables: {top_50_syl_ids}")

    # Second pass: encode with top 50 syllables
    kpt_moseq_dict = {}
    with h5py.File(path, "r") as f:
        for animal_id in tqdm(sorted(f.keys()), desc="One-hot encoding"):
            grp = f[animal_id]
            if "syllable" not in grp:
                continue
            
            syllables = np.asarray(grp["syllable"], dtype=np.int64)
            if syllables.ndim > 1:
                syllables = syllables.flatten()
            
            # One-hot encode only top 50 syllables
            one_hot = np.zeros((len(syllables), 50), dtype=np.float32)
            for t, syl_id in enumerate(syllables):
                syl_id = int(syl_id)
                if syl_id in syl_to_idx:
                    one_hot[t, syl_to_idx[syl_id]] = 1.0
                # else: keep as all zeros for unknown syllables
            
            kpt_moseq_dict[animal_id] = one_hot
    
    return kpt_moseq_dict



def _fit_frame_pca_if_needed(train_dict, args):
    """
    Fit PCA on frame-level features (before windowing) to avoid high-memory flatten windows.
    """
    if args.agg_method != "flatten":
        return None

    # Infer frame feature dim from first valid train run.
    frame_dim = None
    for run_id in sorted(train_dict.keys()):
        frame_feats = _build_frame_feature_matrix(train_dict[run_id])
        if frame_feats.size > 0 and frame_feats.shape[1] > 0:
            frame_dim = frame_feats.shape[1]
            break

    if frame_dim is None or frame_dim <= 50:
        return None

    pca = IncrementalPCA(n_components=50, batch_size=8192)
    fitted_any = False
    for run_id in tqdm(sorted(train_dict.keys())):
        frame_feats = _build_frame_feature_matrix(train_dict[run_id])
        if frame_feats.shape[0] == 0:
            continue
        pca.partial_fit(frame_feats)
        fitted_any = True

    if not fitted_any:
        return None

    print(f"Fitted pre-window PCA: {frame_dim} -> 50 dims")
    return pca



def _build_frame_feature_matrix(run_kinematics):
    # Keypoints can be stored directly as an array per run, e.g. [L, 27, 2].
    if isinstance(run_kinematics, np.ndarray):
        arr = np.asarray(run_kinematics)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim >= 2:
            arr = arr.reshape(arr.shape[0], -1)

        arr = arr.astype(np.float32, copy=False)
        arr[~np.isfinite(arr)] = np.nan
        col_means = np.nanmean(arr, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        nan_rows, nan_cols = np.where(np.isnan(arr))
        arr[nan_rows, nan_cols] = col_means[nan_cols]
        return arr

    feat_blocks = []
    min_len = None

    for values in run_kinematics.values():
        arr = np.asarray(values)
        if arr.ndim == 0:
            continue
        if arr.ndim == 1:
            arr = arr[:, None]
        else:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[0] == 0:
            continue

        min_len = arr.shape[0] if min_len is None else min(min_len, arr.shape[0])
        feat_blocks.append(arr)

    if not feat_blocks or min_len is None or min_len <= 0:
        return np.empty((0, 0), dtype=np.float32)

    feat_blocks = [blk[:min_len] for blk in feat_blocks]
    frame_feats = np.concatenate(feat_blocks, axis=1).astype(np.float32, copy=False)

    frame_feats[~np.isfinite(frame_feats)] = np.nan
    col_means = np.nanmean(frame_feats, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_rows, nan_cols = np.where(np.isnan(frame_feats))
    frame_feats[nan_rows, nan_cols] = col_means[nan_cols]

    return frame_feats


def _window_sequence(frame_feats, window_size, stride, agg_method):
    if frame_feats.shape[0] < window_size:
        return np.empty((0, frame_feats.shape[1]), dtype=np.float32)

    windows = []
    for start in range(0, frame_feats.shape[0] - window_size + 1, stride):
        window = frame_feats[start : start + window_size]
        if agg_method == "mean":
            windows.append(np.nanmean(window, axis=0))
        elif agg_method == "median":
            windows.append(np.nanmedian(window, axis=0))
        elif agg_method == "flatten":
            flat_window = window.reshape(-1)
            # Any remaining NaNs are already column-imputed upstream, but keep this safe.
            if np.isnan(flat_window).any():
                valid = np.where(~np.isnan(flat_window))[0]
                if valid.size > 0:
                    missing = np.where(np.isnan(flat_window))[0]
                    flat_window[missing] = np.interp(missing, valid, flat_window[valid])
                else:
                    flat_window = np.zeros_like(flat_window)
            windows.append(flat_window)
        else:
            raise ValueError(
                f"Unknown agg_method '{agg_method}'. Use one of: mean, median, flatten"
            )

    return np.asarray(windows, dtype=np.float32)


def _concat_windows_for_prediction(run_windows, n_windows_per_prediction):
    if n_windows_per_prediction < 1:
        raise ValueError("n_windows_per_prediction must be >= 1")

    if run_windows.shape[0] < n_windows_per_prediction:
        return np.empty((0, run_windows.shape[1] * n_windows_per_prediction), dtype=np.float32)

    n_groups = run_windows.shape[0] // n_windows_per_prediction
    usable = n_groups * n_windows_per_prediction
    grouped = run_windows[:usable].reshape(n_groups, n_windows_per_prediction, run_windows.shape[1])
    return grouped.reshape(n_groups, -1)


def _build_windowed_split(
    kinematics_dict,
    meta_data,
    window_size,
    stride,
    agg_method,
    n_windows_per_prediction,
    pre_window_pca=None,
):
    if "animal_id" not in meta_data.columns or "strain" not in meta_data.columns:
        raise ValueError("metadata file must contain 'animal_id' and 'strain' columns")

    strain_map = dict(
        zip(meta_data["animal_id"].astype(str), meta_data["strain"].astype(str))
    )

    xs, ys = [], []

    for run_id in sorted(kinematics_dict.keys()):
        run_kinematics = kinematics_dict[run_id]
        animal_id, exp, age = _extract_run_labels(run_id)
        strain = strain_map.get(animal_id, "unknown_strain")

        frame_feats = _build_frame_feature_matrix(run_kinematics)
        if pre_window_pca is not None and frame_feats.shape[0] > 0:
            frame_feats = pre_window_pca.transform(frame_feats).astype(np.float32, copy=False)
        run_windows = _window_sequence(frame_feats, window_size, stride, agg_method)
        run_samples = _concat_windows_for_prediction(run_windows, n_windows_per_prediction)
        if run_samples.shape[0] == 0:
            continue
            
        xs.append(run_samples)
        ys.append(
            np.tile(
                np.array([[age, exp, strain]], dtype=object),
                (run_samples.shape[0], 1),
            )
        )

    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 3), dtype=object)

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def _build_run_metadata(data_dict, meta_data):
    if "animal_id" not in meta_data.columns or "strain" not in meta_data.columns:
        raise ValueError("metadata file must contain 'animal_id' and 'strain' columns")

    strain_map = dict(
        zip(meta_data["animal_id"].astype(str), meta_data["strain"].astype(str))
    )

    rows = []
    for run_id in sorted(data_dict.keys()):
        animal_id, exp, age = _extract_run_labels(run_id)
        rows.append(
            {
                "run_id": run_id,
                "animal_id": animal_id,
                "age": str(age),
                "exp": str(exp),
                "strain": str(strain_map.get(animal_id, "unknown_strain")),
            }
        )

    run_df = pd.DataFrame(rows)
    if run_df.empty:
        raise ValueError("No runs available to create train/test splits")
    return run_df


def _select_balanced_test_runs(run_df, seed):
    """
    Select test runs by randomly picking one animal per strain.
    All runs from those selected animals form the test set.
    """
    strains = sorted(run_df["strain"].unique().tolist())
    rng = np.random.default_rng(seed)

    selected_animals = []
    for strain in strains:
        strain_animals = run_df[run_df["strain"] == strain]["animal_id"].unique()
        if len(strain_animals) == 0:
            raise ValueError(f"No animals found for strain: {strain}")
        # Randomly pick one animal from this strain
        selected_animal = rng.choice(strain_animals)
        selected_animals.append(selected_animal)

    # Get all runs from selected animals
    test_run_ids = run_df[run_df["animal_id"].isin(selected_animals)]["run_id"].tolist()
    return test_run_ids


def _split_train_for_generated_test(train_dict, meta_data, seed):
    run_df = _build_run_metadata(train_dict, meta_data)
    test_run_ids = set(_select_balanced_test_runs(run_df, seed))
    if not test_run_ids:
        raise ValueError("Generated empty test split")

    train_split = {k: v for k, v in train_dict.items() if k not in test_run_ids}
    test_split = {k: v for k, v in train_dict.items() if k in test_run_ids}

    if not train_split:
        raise ValueError("Generated test split consumed all runs, leaving empty train split")

    split_df = run_df[run_df["run_id"].isin(test_run_ids)]
    print(
        "Generated test split from train data: "
        f"{len(test_split)} runs, "
        f"strains={split_df['strain'].nunique()}, "
        f"ages={split_df['age'].value_counts().to_dict()}, "
        f"experimental_stages={split_df['exp'].value_counts().to_dict()}"
    )

    return train_split, test_split


def load_data(args):
    meta_data = pd.read_csv(args.metadata_path, sep="\t")
    has_test_path = bool(args.test_dataset_path) and os.path.exists(args.test_dataset_path)

    if args.dataset == "kinematics":
        train = np.load(args.train_dataset_path, allow_pickle=True)["kinematics"].item()
        if has_test_path:
            test = np.load(args.test_dataset_path, allow_pickle=True)["kinematics"].item()
        else:
            train, test = _split_train_for_generated_test(train, meta_data, args.seed)
    elif args.dataset == "embeddings":
        train_ext = os.path.splitext(args.train_dataset_path)[1].lower()

        if train_ext in {".h5", ".hdf5"}:
            train = _load_embeddings_h5(args.train_dataset_path, args.embedding_stage)
        else:
            train = np.load(args.train_dataset_path, allow_pickle=True)["embeddings"].item()

        if has_test_path:
            test_ext = os.path.splitext(args.test_dataset_path)[1].lower()
            if test_ext in {".h5", ".hdf5"}:
                test = _load_embeddings_h5(args.test_dataset_path, args.embedding_stage)
            else:
                test = np.load(args.test_dataset_path, allow_pickle=True)["embeddings"].item()
        else:
            train, test = _split_train_for_generated_test(train, meta_data, args.seed)
    elif args.dataset == "keypoints":
        train = np.load(args.train_dataset_path, allow_pickle=True)["keypoints"].item()
        if has_test_path:
            test = np.load(args.test_dataset_path, allow_pickle=True)["keypoints"].item()
        else:
            train, test = _split_train_for_generated_test(train, meta_data, args.seed)
    elif args.dataset == "kp-moseq":
        train = _load_kpt_moseq(args.train_dataset_path)
        if has_test_path:
            test = _load_kpt_moseq(args.test_dataset_path)
        else:
            train, test = _split_train_for_generated_test(train, meta_data, args.seed)
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Use one of: 'kinematics', 'embeddings', 'keypoints', 'kp-moseq'")

    pre_window_pca = _fit_frame_pca_if_needed(train, args)
    print("Raw data loaded")
    X_train, y_train = _build_windowed_split(
        kinematics_dict=train,
        meta_data=meta_data,
        window_size=args.window_size,
        stride=args.window_stride,
        agg_method=args.agg_method,
        n_windows_per_prediction=args.n_windows_per_prediction,
        pre_window_pca=pre_window_pca,
    )
    X_test, y_test = _build_windowed_split(
        kinematics_dict=test,
        meta_data=meta_data,
        window_size=args.window_size,
        stride=args.window_stride,
        agg_method=args.agg_method,
        n_windows_per_prediction=args.n_windows_per_prediction,
        pre_window_pca=pre_window_pca,
    )
    print(f"Train samples: {X_train.shape[0]} and Test samples: {X_test.shape[0]} with shape {X_train.shape}")

    return X_train, X_test, y_train, y_test


def get_model(args):
    if args.classifier_model_type == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=args.seed)
    if args.classifier_model_type =="dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Unknown classifier model type: {args.classifier_model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Classification of the Arena Dataset", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    set_reproducibility(args.seed)

    kinematics_train, kinematics_test, label_train, label_test = load_data(args)
    if kinematics_train.shape[0] == 0 or kinematics_test.shape[0] == 0:
        raise ValueError("No windowed samples produced. Check data and window arguments.")

    # Create dated output directory.
    run_dir = os.path.join(args.output_dir, datetime.now().strftime("%Y-%m-%d-H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Save args.
    args_path = os.path.join(run_dir, "args.txt")
    with open(args_path, "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
    print(f"Args saved to {args_path}")

    model = get_model(args)

    for i, label in enumerate(["age", "experimental_phase", "strain"]):
        enc = LabelEncoder()
        y_train = enc.fit_transform(label_train[:, i])
        y_test = enc.transform(label_test[:, i])

        # stratify data
        rus = RandomUnderSampler(random_state=args.seed)
        kinematics_train_us, y_train_us = rus.fit_resample(kinematics_train, y_train)
        kinematics_test_us, y_test_us = rus.fit_resample(kinematics_test, y_test)

        model.fit(kinematics_train_us, y_train_us)
        pred = model.predict(kinematics_test_us)
        report = classification_report(y_test_us, pred, target_names=enc.classes_, zero_division=0.0)

        report_path = os.path.join(run_dir, f"report_{label}.txt")
        with open(report_path, "w") as f:
            f.write(f"Classification report for {label}:\n")
            f.write(report)
        print(f"Report saved to {report_path}")
