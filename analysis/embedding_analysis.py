import argparse
import gc
import os

from analysis.embeds_to_pose import embeds_to_pose
from analysis.utils.window_and_aggregate import window_and_aggregate_all_layers
import numpy as np
import matplotlib.pyplot as plt

from .preprocessing.load_data import load_data
from .preprocessing.apply_pca import apply_pca
from .cluster.cluster import cluster_analysis
from .manifold_analysis import manifold_analysis
from .modeling import modeling
from .utils.save_args import save_args
from .utils.title_print import title_print

def get_args_parser():
    parser = argparse.ArgumentParser("Embedding Analysis", add_help=False)
    parser.add_argument(
        "--dataset",
        default="arena",
        help="dataset name (arena, shot7m2, mabe_mice, hbabel)",
    )
    parser.add_argument(
        "--embed_type",
        default="windowed",
        help="type of embeddings to load (windowed, full_sequence, ts2vec)",
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
        "--ndim_for_cluster",
        default=2,
        type=int,
        help="number of dimensions to keep for the cluster silouhette analysis",
    )
    parser.add_argument(
        "--silhouette_ds_rate",
        default=1,
        type=int,
        help="downsampling rate for silhouette score calculation (to avoid long compute time)",
    )
    parser.add_argument(
        "--kmeans_k_range",
        default=(2, 11),
        nargs=2,
        type=int,
        help="range of k values for KMeans clustering",
    )
    parser.add_argument(
        "--kmeans_ks",
        default=[2, 3, 4],
        nargs="+",
        type=int,
        help="list of k values per layer for KMeans clustering",
    )
    parser.add_argument(
        "--raw_kmeans_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of k values per layer for KMeans clustering on raw embeddings",
    )
    parser.add_argument(
        "--umap_kmeans_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of k values per layer for KMeans clustering on UMAP embeddings",
    )
    parser.add_argument(
        "--tsne_kmeans_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of k values per layer for KMeans clustering on TSNE embeddings",
    )
    parser.add_argument(
        "--hdbscan_cluster_size_range",
        default=(10, 5000, 50),
        nargs=3,
        type=int,
        help="range of cluster size values for HDBSCAN clustering",
    )
    parser.add_argument(
        "--raw_hdbscan_cluster_size",
        default=None,
        nargs="+",
        type=int,
        help="list of cluster size values per layer for HDBSCAN clustering on raw embeddings",
    )
    parser.add_argument(
        "--umap_hdbscan_cluster_size",
        default=None,
        nargs="+",
        type=int,
        help="list of cluster size values per layer for HDBSCAN clustering on UMAP embeddings",
    )
    parser.add_argument(
        "--tsne_hdbscan_cluster_size",
        default=None,
        nargs="+",
        type=int,
        help="list of cluster size values per layer for HDBSCAN clustering on TSNE embeddings",
    )
    parser.add_argument(
        "--gmm_ks_range",
        default=(2, 11),
        nargs=2,
        type=int,
        help="range of number of components per layer for Pyro GMM clustering",
    )
    parser.add_argument(
        "--gmm_ks",
        default=[],
        nargs="+",
        type=int,
        help="list of number of components per layer for Pyro GMM clustering",
    )
    parser.add_argument(
        "--raw_gmm_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of number of components per layer for Pyro GMM clustering on raw embeddings",
    )
    parser.add_argument(
        "--umap_gmm_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of number of components per layer for Pyro GMM clustering on UMAP embeddings",
    )
    parser.add_argument(
        "--tsne_gmm_ks",
        default=None,
        nargs="+",
        type=int,
        help="list of number of components per layer for Pyro GMM clustering on TSNE embeddings",
    )
    parser.add_argument(
        "--gmm_max_iter",
        default=100,
        type=int,
        help="maximum number of EM iterations for Pyro GMM",
    )
    parser.add_argument(
        "--gmm_tol",
        default=1e-3,
        type=float,
        help="log-likelihood convergence tolerance for Pyro GMM",
    )
    parser.add_argument(
        "--gmm_reg_covar",
        default=1e-6,
        type=float,
        help="diagonal covariance floor for Pyro GMM",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for reproducibility of results (clustering, train/test splits, etc.)",
    )
    parser.add_argument(
        "--pca_dims",
        default=70,
        type=int,
        help="Minimum number of dimensions when pca is applied to reduce dimensionality",
    )
    parser.add_argument(
        "--pca_batch_size",
        default=50000,
        type=int,
        help="Batch size used for IncrementalPCA to reduce peak memory usage",
    )
    parser.add_argument(
        "--agg_method",
        default="mean",
        help="method for aggregating tokens within a window (mean, max, first, last)"
    )
    parser.add_argument(
        "--n_windows_per_run_for_metadata",
        default=-1,
        type=int,
        help="number of windows to consider per run for metadata classification"
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
    parser.add_argument(
        "--sample",
        default=False,
        action="store_true",
        help="whether to run the analysis on a small subsample of the data for quick testing",
    )
    parser.add_argument(
        "--base_window_size",
        default=100,
        type=int,
        help="base window size for token aggregation (used as reference for other window sizes)",
    )
    parser.add_argument(
        "--base_window_stride",
        default=100,
        type=int,
        help="base window stride for token aggregation (used as reference for other window strides)",
    )
    parser.add_argument(
        "--n_windows_per_run_base",
        default=2,
        type=int,
        help="number of windows to consider per run for the base windowing (if -1, consider all windows)",
    )
    parser.add_argument(
        "--n_windows_per_run_for_clustering",
        default=10,
        type=int,
        help="number of windows to consider per run for clustering analysis (if -1, consider all windows)",
    )

    return parser


def embedding_analysis(args):
    # output path needs to separate per layer and full sequence embedding extraction
    args.output_dir = os.path.join(args.output_dir, args.embed_type, f"ws-{args.base_window_size}")
    os.makedirs(args.output_dir, exist_ok=True)
    title_print("Loading Data")

    embeddings, raw_token_shapes, metadata, keypoints, kinematics, syllable_labels, true_labels, true_label_names = load_data(args)
    title_print("Applying PCA")
    apply_pca(
        embeddings,
        n_components=args.pca_dims,
        output_dir=args.output_dir,
        batch_size=args.pca_batch_size,
        seed=args.seed,
    )

    embeddings_windowed, raw_layer_window_maps, windowed_token_shapes = window_and_aggregate_all_layers(
        embeddings,
        raw_token_shapes,
        window_size=args.base_window_size,
        stride=args.base_window_stride,
        agg_method=args.agg_method,
        n_windows_per_run=args.n_windows_per_run_base,
        seed=args.seed,
    )
    title_print("Raw Embeddings Analysis")
    title_print("Cluster Analysis", n=10)

    raw_clustroids_idxs = cluster_analysis(args=args, embeddings=embeddings_windowed, 
                                           token_shapes=windowed_token_shapes, true_labels=true_labels, 
                                           true_label_names=true_label_names, metadata=metadata, kinematics=kinematics, 
                                           syllable_labels=syllable_labels, data_type="raw")
    embeds_to_pose(embeddings=embeddings, clustroids_idxs=raw_clustroids_idxs, layer_window_maps=raw_layer_window_maps, 
                   window_size=args.base_window_size, token_shapes=raw_token_shapes, keypoints=keypoints, output_dir=args.output_dir, data_type="raw")
    
    title_print("Modeling", n=10)
    modeling(embeddings=embeddings_windowed, metadata=metadata, kinematics=kinematics, syllable_labels=syllable_labels, 
             windowed_token_shapes=windowed_token_shapes, raw_token_shapes=raw_token_shapes, args=args, 
             layer_window_maps=raw_layer_window_maps, data_type="raw")

    title_print("Manifold to UMAP and TSNE")
    # Manifold extraction (TSNE, UMAP) and visualization of the embeddings colored by metadata variables (e.g. strain)
    tsne_embeddings, umap_embeddings = manifold_analysis(embeddings=embeddings_windowed, args=args, token_shapes=windowed_token_shapes)
    # Raw windowed embeddings are no longer needed after manifold embeddings are produced.
    del embeddings_windowed
    gc.collect()

    title_print("UMAP Embeddings Analysis")
    title_print("Cluster Analysis", n=10)
    umap_clustroids_idxs = cluster_analysis(args=args, embeddings=umap_embeddings, token_shapes=windowed_token_shapes, true_labels=true_labels, true_label_names=true_label_names, metadata=metadata, kinematics=kinematics, syllable_labels=syllable_labels, data_type="umap")
    embeds_to_pose(embeddings=embeddings, clustroids_idxs=umap_clustroids_idxs, layer_window_maps=raw_layer_window_maps, window_size=args.base_window_size, token_shapes=raw_token_shapes, keypoints=keypoints, output_dir=args.output_dir, data_type="umap")
    title_print("Modeling", n=10)
    modeling(embeddings=umap_embeddings, metadata=metadata, kinematics=kinematics, syllable_labels=syllable_labels, windowed_token_shapes=windowed_token_shapes, raw_token_shapes=raw_token_shapes, args=args, layer_window_maps=raw_layer_window_maps, data_type="umap")

    # UMAP embeddings are no longer needed after UMAP analysis.
    del umap_embeddings
    gc.collect()

    title_print("TSNE Embeddings Analysis")
    title_print("Cluster Analysis", n=10)
    tsne_clustroids_idxs = cluster_analysis(args=args, embeddings=tsne_embeddings, token_shapes=windowed_token_shapes, true_labels=true_labels, true_label_names=true_label_names, metadata=metadata, kinematics=kinematics, syllable_labels=syllable_labels, data_type="tsne")
    embeds_to_pose(embeddings=embeddings, clustroids_idxs=tsne_clustroids_idxs, layer_window_maps=raw_layer_window_maps, window_size=args.base_window_size, token_shapes=raw_token_shapes, keypoints=keypoints, output_dir=args.output_dir, data_type="tsne")
    title_print("Modeling", n=10)
    modeling(embeddings=tsne_embeddings, metadata=metadata, kinematics=kinematics, syllable_labels=syllable_labels, windowed_token_shapes=windowed_token_shapes, raw_token_shapes=raw_token_shapes, args=args, layer_window_maps=raw_layer_window_maps, data_type="tsne")

    # TSNE embeddings and shared window maps are no longer needed.
    del tsne_embeddings, raw_layer_window_maps
    gc.collect()

    # Decoding Analysis

    # Save args as json
    save_args(args, args.output_dir)
    title_print("Analysis Complete", n=50)