from datasets.keypoints import get_kinematics
import numpy as np

if __name__ == "__main__":
    keypoins_path = '/scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz'
    keypoints = np.load(keypoins_path, allow_pickle=True)['keypoints'].item()
    kinematics = get_kinematics(keypoints)
    np.savez(keypoins_path.replace('shuffle-3_split', 'kinematics'), kinematics=kinematics)