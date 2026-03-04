from pathlib import Path
from torchvision import transforms
import numpy as np
import pickle
import torch

from .pose_traj_dataset import BasePoseTrajDataset
from .augmentations import GaussianNoise, Rotation, Reflect


class ArenaDataset(BasePoseTrajDataset):
    """
    Open-field wandering of single mice in an arena.
    """

    DEFAULT_FRAME_RATE = 50 # To check
    DEFAULT_GRID_SIZE = 250
    NUM_KEYPOINTS = 27
    KPTS_DIMENSIONS = 2
    NUM_INDIVIDUALS = 1
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS)
    SAMPLE_LEN = 1800

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
        augmentations: transforms.Compose = None,
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

        self.preprocess()

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

    def featurise_keypoints(self, keypoints):
        keypoints = self.normalize(keypoints)
        if self.centeralign:
            keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            keypoints = self.transform_to_centeralign_components(keypoints, center_idx=self.BODY_PART_2_INDEX["mouse_center"])
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints
    
    def __getitem__(self, idx: int):

        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0]][
            subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len
        ]
        inputs = self.prepare_subsequence_sample(subsequence)
        return inputs, []