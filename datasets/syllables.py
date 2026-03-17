import numpy as np
import h5py
from tqdm import tqdm

def load_kpt_moseq(path, one_hot_encode=False):
    """
    Loads kpt-moseq data from an H5 file with structure:
      /<animal_id>/syllable -> ndarray
    Keeps only the 50 most common syllables.
    If one_hot_encode=True: returns (n_frames, 50) one-hot arrays, unknown syllables → all zeros.
    If one_hot_encode=False: returns (n_frames,) integer arrays with original syllable IDs,
                             unknown syllables (not in top 50) → -1.
    """
    # First pass: count syllable frequencies
    syllable_counts = {}
    with h5py.File(path, "r") as f:
        for animal_id in tqdm(sorted(f.keys()), desc="Counting syllables"):
            grp = f[animal_id]
            if "syllable" not in grp:
                continue
            syllables = np.asarray(grp["syllable"], dtype=np.int64).flatten()
            for syl in syllables:
                syl_id = int(syl)
                syllable_counts[syl_id] = syllable_counts.get(syl_id, 0) + 1

    # Get top 50 most common syllables
    top_50_syllables = sorted(syllable_counts.items(), key=lambda x: x[1], reverse=True)[:50]
    top_50_syl_ids = [syl_id for syl_id, _ in top_50_syllables]
    syl_to_idx = {syl_id: idx for idx, syl_id in enumerate(top_50_syl_ids)}

    print(f"Using top 50 syllables: {top_50_syl_ids}")

    # Second pass: encode
    desc = "One-hot encoding" if one_hot_encode else "Loading syllables"
    kpt_moseq_dict = {}
    with h5py.File(path, "r") as f:
        for animal_id in tqdm(sorted(f.keys()), desc=desc):
            grp = f[animal_id]
            if "syllable" not in grp:
                continue

            syllables = np.asarray(grp["syllable"], dtype=np.int64).flatten()

            if one_hot_encode:
                encoded = np.zeros((len(syllables), 50), dtype=np.float32)
                for t, syl_id in enumerate(syllables):
                    if int(syl_id) in syl_to_idx:
                        encoded[t, syl_to_idx[int(syl_id)]] = 1.0
            else:
                # Map to contiguous indices 0–49, unknown → -1
                encoded = np.array(
                    [syl_to_idx.get(int(syl), -1) for syl in syllables],
                    dtype=np.int64
                )

            kpt_moseq_dict[animal_id] = encoded

    return kpt_moseq_dict