import os
import numpy as np
import pandas as pd
from datasets.keypoints import get_kinematics
from datasets.syllables import load_kpt_moseq
from datasets.arena_dataset import extract_metadata_from_runid

def load_data(args):
    if args.dataset == "arena":
        # Load embeddings
        embedding_file = os.path.join(args.path_to_emb_dir, "embeddings.npy")
        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embeddings file not found: {embedding_file}. There was probably an error during the embedding extraction step.")
        embeddings = np.load(embedding_file, allow_pickle=True).item()
        token_shapes_file = os.path.join(args.path_to_emb_dir, "token_shapes.npy")
        if not os.path.exists(token_shapes_file):
            raise FileNotFoundError(f"Token shapes file not found: {token_shapes_file}. There was probably an error during the embedding extraction step.")
        token_shapes = np.load(token_shapes_file, allow_pickle=True)

        run_ids = list(embeddings[next(iter(embeddings))].keys())
        print(f"Loaded embeddings for {len(embeddings)} layers, {len(run_ids)} runs per layer")

        # get metadata from run_ids
        metadata = pd.DataFrame([extract_metadata_from_runid(run_id) for run_id in run_ids])
 

        # Load metadata
        if args.meta_data_path is not None:
            if not os.path.exists(args.meta_data_path):
                raise FileNotFoundError(f"Metadata file not found: {args.meta_data_path}")
            strain_metadata = pd.read_csv(args.meta_data_path, sep="\t")
            # add a column strain based on the animal id and the known mapping of animal id to strain in metadata
            metadata = metadata.merge(strain_metadata[["animal_id", "strain"]], left_on="animal_id", right_on="animal_id", how="left")

        print(f"Extracted metadata: {metadata.columns.tolist()}")

        # Load keypoints
        if args.keypoints_path is not None:
            if not os.path.exists(args.keypoints_path):
                raise FileNotFoundError(f"Keypoints file not found: {args.keypoints_path}")
            keypoints = np.load(args.keypoints_path, allow_pickle=True)['keypoints'].item()
            kinematics = get_kinematics(keypoints, mean=False)
            print(f"Extracted kinematics: {kinematics[list(kinematics.keys())[0]].keys()}")
        else:
            kinematics = None

        # Load syllable labels
        if args.syllable_labels_path is not None:
            if not os.path.exists(args.syllable_labels_path):
                raise FileNotFoundError(f"Syllable labels file not found: {args.syllable_labels_path}")
            syllable_labels = load_kpt_moseq(args.syllable_labels_path, one_hot_encode=False)
        else:
            syllable_labels = None
            

    return embeddings, token_shapes, metadata, kinematics, syllable_labels