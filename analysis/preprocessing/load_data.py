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

    elif args.dataset == "shot7m2":
        if args.per_layer_embeddings:
            # get all files with the shape "test_submission_*.npy"
            embedding_files = [f for f in os.listdir(args.path_to_emb_dir) if f.startswith("test_submission_") and f.endswith(".npy")]
            if len(embedding_files) == 0:
                raise FileNotFoundError(f"No embedding files found in {args.path_to_emb_dir} with the pattern 'test_submission_*.npy'. There was probably an error during the embedding extraction step.")
            embeddings = {f"layer_{i}": {} for i in range(len(embedding_files))}  # initialize embeddings dict with layer keys
            token_shapes = []
            for i, embedding_file in enumerate(embedding_files):
                layer_key = f"layer_{i}"
                embedding_path = os.path.join(args.path_to_emb_dir, embedding_file)
                data = np.load(embedding_path, allow_pickle=True).item()
                embed = data['embeddings']
                frame_number_map = data['frame_number_map']
                # re order embeddings per run in frame_number_map 
                for run in frame_number_map.keys():
                    frame_numbers = frame_number_map[run]
                    # frame numbers are in the shape (start, end) and we want to select the embeddings in that range
                    frame_idx = np.arange(frame_numbers[0], frame_numbers[1])
                    embeddings[layer_key][run] = embed[frame_idx]

                # create a fake token_shapes because everything is already at the frame level
                token_shapes.append((1,1,1))
            sequence_names = list(embeddings[next(iter(embeddings))].keys())
            metadata = None
            kinematics = None
            syllable_labels = None
            print(f"Loaded embeddings for {len(embeddings)} layers, {len(sequence_names)} sequences per layer")
        else:
            # Load embeddings
            embedding_file = os.path.join(args.path_to_emb_dir, "embeddings.npy")
            if not os.path.exists(embedding_file):
                raise FileNotFoundError(f"Embeddings file not found: {embedding_file}. There was probably an error during the embedding extraction step.")
            embeddings = np.load(embedding_file, allow_pickle=True).item()
            token_shapes_file = os.path.join(args.path_to_emb_dir, "token_shapes.npy")
            if not os.path.exists(token_shapes_file):
                raise FileNotFoundError(f"Token shapes file not found: {token_shapes_file}. There was probably an error during the embedding extraction step.")
            token_shapes = np.load(token_shapes_file, allow_pickle=True)
            sequence_names = list(embeddings[next(iter(embeddings))].keys())

            # for shot7M2 let's just return the sequence names as metadata, since we don't have any other metadata for this dataset
            metadata = None
            kinematics = None
            syllable_labels = None

            # select only the first two runs 
            for layer in embeddings.keys():
                selected_runs = sequence_names[:2]
                embeddings[layer] = {run: embeddings[layer][run] for run in selected_runs}
            print(f"Loaded embeddings for {len(embeddings)} layers, {len(sequence_names)} sequences per layer")

            
    return embeddings, token_shapes, metadata, kinematics, syllable_labels