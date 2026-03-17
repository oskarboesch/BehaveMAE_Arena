import argparse

from .preprocessing.load_data import load_data
from .manifold_analysis import manifold_analysis
from .modeling import modeling
from .utils.save_args import save_args

def get_args_parser():
    parser = argparse.ArgumentParser("Embedding Analysis", add_help=False)
    parser.add_argument(
        "--dataset",
        default="arena",
        help="dataset name (arena, shot7m2, mabe_mice, hbabel)",
    )
    parser.add_argument(
        "--path_to_emb_dir",
        default="/scratch/izar/boesch/BehaveMAE/outputs/arena/experiment5",
        help="path where to load data from",
    )
    parser.add_argument(
        "--meta_data_path",
        default=None,
        help="path to metadata TSV file",
    )
    parser.add_argument(
        "--keypoints_path",
        default=None,
        help="path to keypoints numpy file",
    )
    parser.add_argument(
        "--syllable_labels_path",
        default=None,
        help="path to syllable labels numpy file",
    )
    parser.add_argument(
        "--output_dir",
        default="/scratch/izar/boesch/BehaveMAE/outputs/arena/experiment5/emb_analysis",
        help="path where to save",
    )
    parser.add_argument(
        "--min_vids_per_strain",
        default=30,
        type=int,
        help="minimum number of videos necessary to include a strain in the analysis (to avoid class imbalance issues)",
    )
    parser.add_argument(
        "--ndim_for_cluster_silhouette",
        default=50,
        type=int,
        help="number of dimensions to keep for the cluster silouhette analysis",
    )
    parser.add_argument(
        "--agg_method",
        default="mean",
        help="method for aggregating tokens within a window (mean, max, first, last)"
    )
    parser.add_argument(
        "--window_size_manifold",
        default=100,
        type=int,
        help="size of the window for token aggregation during manifold calculation"
    )
    parser.add_argument(
        "--window_stride_manifold",
        default=100,
        type=int,
        help="stride of the window for token aggregation during manifold calculation"
    )
    parser.add_argument(
        "--window_size_metadata",
        default=100,
        type=int,
        help="size of the window for token aggregation during metadata classification"
    )
    parser.add_argument(
        "--window_stride_metadata",
        default=100,
        type=int,
        help="stride of the window for token aggregation during metadata classification"
    )
    parser.add_argument(
        "--n_windows_per_run_for_metadata",
        default=10,
        type=int,
        help="number of windows to consider per run for metadata classification"
    )
    parser.add_argument(
        "--window_size_syllables",
        default=25,
        type=int,
        help="size of the window for token aggregation during syllable classification"
    )
    parser.add_argument(
        "--window_stride_syllables",
        default=25,
        type=int,
        help="stride of the window for token aggregation during syllable classification"
    )
    parser.add_argument(
        "--max_n_samples_for_syllable_analysis",
        default=10000,
        type=int,
        help="maximum number of samples to use for syllable analysis (to avoid long compute time)"
    )
    parser.add_argument(
        "--max_nan_ratio_per_window",
        default=0.2,
        type=float,
        help="maximum ratio of NaN values allowed per window"
    )

    return parser


def embedding_analysis(args):
    embeddings, token_shapes, metadata, kinematics, syllable_labels = load_data(args)

    # Manifold extraction (TSNE, UMAP) and visualization of the embeddings colored by metadata variables (e.g. strain)
    tsne_embeddings, umap_embeddings, layer_window_maps = manifold_analysis(embeddings, args, token_shapes)


    # Linear/logistic regression analysis to predict metadata variables from the embeddings
    modeling(embeddings, metadata, kinematics, syllable_labels, token_shapes, args, data_type="raw")

    # Linear/logistic regression on umap and tsne
    modeling(umap_embeddings, metadata, kinematics, syllable_labels, token_shapes, args, layer_window_maps, data_type="umap")
    modeling(tsne_embeddings, metadata, kinematics, syllable_labels, token_shapes, args, layer_window_maps, data_type="tsne")


    # TODO (PCA if too many dimensins, add clustering analysis, silhouette scores, etc.) 

    # Save args as json
    save_args(args, args.output_dir)