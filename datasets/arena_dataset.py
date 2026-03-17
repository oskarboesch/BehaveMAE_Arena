from pathlib import Path
import numpy as np
import pickle
import torch

from .pose_traj_dataset import BasePoseTrajDataset
from .augmentations import GaussianNoise, Rotation, Reflect
from matplotlib.collections import LineCollection


import matplotlib.pyplot as plt

class ArenaDataset(BasePoseTrajDataset):
    """
    Open-field wandering of single mice in an arena.
    """

    DEFAULT_FRAME_RATE = 25 
    DEFAULT_GRID_SIZE = 500
    NUM_KEYPOINTS = 27
    KPTS_DIMENSIONS = 2
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    DEFAULT_NUM_TESTING_POINTS = 10

    STR_BODY_PARTS = [
        "nose",
        "left_ear",
        "right_ear",
        "left_ear_tip",
        "right_ear_tip",
        "left_eye",
        "right_eye",
        "neck",
        "mid_back",
        "mouse_center",
        "mid_backend",
        "mid_backend2",
        "mid_backend3",
        "tail_base",
        "tail1",
        "tail2",
        "tail3",
        "tail4",
        "tail5",
        "left_shoulder",
        "left_midside",
        "left_hip",
        "right_shoulder",
        "right_midside",
        "right_hip",
        "tail_end",
        "head_midpoint"
    ] 

    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}


    def __init__(
        self,
        mode: str,
        path_to_data_dir: Path,
        scale: bool = True,
        sampling_rate: int = 1,
        num_frames: int = 80,
        sliding_window: int = 1,
        augmentations: "transforms.Compose" = None,
        include_testdata: bool = False,
        split_tokenization: bool = False,
        centeralign: bool = False,
        **kwargs
    ):
        super().__init__(
            path_to_data_dir, scale, sampling_rate, num_frames, sliding_window, **kwargs
        )
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.centeralign = centeralign
        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed

        if augmentations:
            from torchvision import transforms

            gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
            self.augmentations = transforms.Compose(
                [
                    Rotation(grid_size=gs, p=0.5),
                    GaussianNoise(p=0.5),
                    Reflect(grid_size=gs, p=0.5),
                ]
            )
        else:
            self.augmentations = None

        self.load_data(include_testdata)

        self.split_tokenization = split_tokenization
        if self.mode == "pretrain" or self.mode == "test":
            self.preprocess()
        elif self.mode == "inference":
            for sequence in self.sequences:
                self.interpolate_nans(sequence)
            # check no nans           
            for sequence in self.sequences:
                assert not np.isnan(sequence).any(), "Inference mode does not allow NaN values in the data."

    def load_data(self, include_testdata) -> None:
        """Loads dataset from npz file containing keypoints dict and confidence dict ."""
        if self.mode == "pretrain":
            # Load npz file

            keypoints_dict = np.load(self.path, allow_pickle=True)['keypoints'].item()

            print(f"Loaded training data from {self.path}. Number of sequences: {len(keypoints_dict)}")

            # Convert dict to list for faster indexing
            self.sequence_names = list(keypoints_dict.keys())
            self.sequences = list(keypoints_dict.values())
            
            if include_testdata:
                test_path = str(self.path).replace("train", "test")
    
                test_keypoints_dict = np.load(test_path, allow_pickle=True)['keypoints'].item()
                self.sequence_names.extend(list(test_keypoints_dict.keys()))
                self.sequences.extend(list(test_keypoints_dict.values()))
        elif self.mode == "test":
            test_path = str(self.path).replace("train", "test")
            keypoints_dict = np.load(test_path, allow_pickle=True)['keypoints'].item()
            self.sequence_names = list(keypoints_dict.keys())
            self.sequences = list(keypoints_dict.values())
                
        elif self.mode == "inference":
            keypoints_dict = np.load(self.path, allow_pickle=True)['keypoints'].item()
            self.sequence_names = list(keypoints_dict.keys())
            self.sequences = list(keypoints_dict.values())
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        print(f"Data loaded from {self.path}. Number of sequences: {len(self.sequences)}")

    def preprocess(self):
        """
        Does initial preprocessing on entire dataset.
        """
        sequences = self.sequences

        seq_keypoints = []
        keypoints_ids = []
        total_windows = 0
        kept_windows = 0
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window

        for seq_ix, vec_seq in enumerate(sequences):
            # Arena data shape: (num_frames, num_keypoints, kpts_dimensions)
            # Expected shape: (num_frames, num_individuals, num_keypoints, kpts_dimensions)
            
            # Add individuals dimension if not present
            if vec_seq.ndim == 3:
                vec_seq = vec_seq[:, np.newaxis, :, :]  # Add individual dimension
            
            vec_seq = vec_seq[:: self.sampling_rate]
            vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
            
            # Pad sequences similar to Shot7M2
            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = 120
            
            pad_vec = np.pad(
                vec_seq,
                ((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)),
                mode="edge",
            )
            
            seq_keypoints.append(pad_vec)
            
            # Store (seq_ix, start_position) tuples, filtering out windows with NaN
            for i in np.arange(0, len(pad_vec) - sub_seq_length + 1, sliding_window):
                total_windows += 1
                window = pad_vec[i : i + sub_seq_length]
                # Only add this window if it contains no NaN values
                if not np.isnan(window).any():
                    keypoints_ids.append((seq_ix, i))
                    kept_windows += 1
            sequences[seq_ix] = None  # Free memory

        del self.sequences

        self.seq_keypoints = np.array(seq_keypoints, dtype=object)
        self.items = list(np.arange(len(keypoints_ids)))
        self.keypoints_ids = keypoints_ids

        self.n_frames = len(self.keypoints_ids)
        
        print(f"Preprocessing complete: kept {kept_windows}/{total_windows} windows without NaN ({100*kept_windows/total_windows:.1f}%)")

    def featurise_keypoints(self, keypoints, windowed=True):
        if self.centeralign:
            # For centeralign, work with unnormalized data first
            # The normalization of center will happen inside transform_to_centeralign_components
            if windowed:
                keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            else:
                keypoints = keypoints.reshape(-1, *self.KEYFRAME_SHAPE)
            keypoints = self.transform_to_centeralign_components(keypoints, center_index=self.BODY_PART_2_INDEX["mouse_center"])
        else:
            # For non-centeralign, normalize the entire vector
            keypoints = self.normalize(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints
    
    def __getitem__(self, idx: int):

        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0]][
            subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]
        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []
    

    def interpolate_nans(self, sequence):
        """Interpolate NaN values in the sequence using linear interpolation."""
        for i in range(sequence.shape[1]):  # Iterate over keypoints
            for j in range(sequence.shape[2]):  # Iterate over dimensions
                keypoint_series = sequence[:, i, j]
                nans = np.isnan(keypoint_series)
                if np.any(nans):
                    not_nans = ~nans
                    if np.sum(not_nans) > 1:  # Need at least two points to interpolate
                        keypoint_series[nans] = np.interp(
                            np.flatnonzero(nans),
                            np.flatnonzero(not_nans),
                            keypoint_series[not_nans],
                        )
        return sequence
    
def plot_kp_in_2d(keypoints, frame_idxs=np.arange(1,50,1), title=None):
    """Plot the keypoints of a specific frame in 2D with skeleton connections."""
    # Define skeleton edges connecting body parts
    skeleton_edges = [
        # Head and ears
        ("left_ear", "head_midpoint"),
        ("right_ear", "head_midpoint"),
        ("head_midpoint", "nose"),
        
        # Spine
        ("nose", "neck"),
        ("neck", "mid_back"),
        ("mid_back", "tail_base"),
        
        # Left side
        ("neck", "left_shoulder"),
        ("left_shoulder", "left_midside"),
        ("left_midside", "left_hip"),
        
        # Right side
        ("neck", "right_shoulder"),
        ("right_shoulder", "right_midside"),
        ("right_midside", "right_hip"),
        
        # Tail
        ("tail_base", "tail1"),
        ("tail1", "tail2"),
        ("tail2", "tail3"),
        ("tail3", "tail4"),
        ("tail4", "tail5"),
        ("tail5", "tail_end"),
        
        # Ears to eyes
        ("left_ear", "left_eye"),
        ("right_ear", "right_eye"),
        ("left_ear_tip", "left_ear"),
        ("right_ear_tip", "right_ear"),
    ]
    
    # Map body part names to indices
    body_part_to_idx = ArenaDataset.BODY_PART_2_INDEX
    
    plt.figure(figsize=(8, 8))
    for i, frame_idx in enumerate(frame_idxs):
        # Plot skeleton connections
        for part1, part2 in skeleton_edges:
            if part1 in body_part_to_idx and part2 in body_part_to_idx:
                idx1 = body_part_to_idx[part1]
                idx2 = body_part_to_idx[part2]
                x = [keypoints[frame_idx, idx1, 0], keypoints[frame_idx, idx2, 0]]
                y = [keypoints[frame_idx, idx1, 1], keypoints[frame_idx, idx2, 1]]
                plt.plot(x, y, 'gray', linewidth=1, alpha=0.6/(3*i+1))
        
        # Plot keypoints on top
        plt.scatter(keypoints[frame_idx, :, 0], keypoints[frame_idx, :, 1], c='blue', s=30, zorder=5, alpha=1/(3*i+1))
    
    plt.title(title if title else f"Frame {frame_idx}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(np.min(keypoints[:, :, 0]), np.max(keypoints[:, :, 0]))
    plt.ylim(np.min(keypoints[:, :, 0]), np.max(keypoints[:, :, 1]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.show()
    

def plot_center_trajectory(keypoints, title=None):
    """Plot the trajectory of the center keypoint over time with a color gradient per frame."""
    
    center_idx = ArenaDataset.BODY_PART_2_INDEX["mouse_center"]

    x = keypoints[:, center_idx, 0]
    y = keypoints[:, center_idx, 1]

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(x))

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_collection(lc)

    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title if title else "Trajectory of Mouse Center")
    ax.grid()
    plt.show()

def extract_metadata_from_runid(run_id):
    """Extracts metadata from the run ID string."""
    parts = run_id.split("_")
    metadata = {
        "run_id": run_id,
        "animal_id": parts[0],
        "experimental_phase": parts[1],
        "age": parts[2],
        "trial_id": parts[3],
    }
    return metadata