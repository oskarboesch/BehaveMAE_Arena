import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def analyze_cluster_labels(cluster_labels, flat_index_map, metadata, kinematics, time_window_size, output_dir, skip_metadata=["strain"]):
    """
    Perform multiple analyses on clustering labels.
    - Analysis 1: Overall Expression Differences Across Metadata (with mean & std)
    - Analysis 2: Temporal Expression Subplots (per run average with SEM error bars)
    - Analysis 3: Mean Kinematics per Cluster (ignoring NaNs)
    - Analysis 4: Cluster Transition Graphs (with colored edges)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Core Data Mapping
    
    # Create DataFrame from index map and labels
    runs, tokens = zip(*flat_index_map)
    df = pd.DataFrame({'run_id': runs, 'token_idx': tokens, 'cluster': cluster_labels})
    
    # Join with metadata
    if metadata is not None and not metadata.empty:
        df = df.merge(metadata, on='run_id', how='left')
    
    unique_clusters = sorted(df['cluster'].unique())
    n_clusters = len(unique_clusters)
    
    # Identify valid metadata columns
    valid_meta_cols = []
    if metadata is not None and not metadata.empty:
        all_meta_cols = [c for c in metadata.columns if c not in ['run_id', 'animal_id', 'trial_id'] + skip_metadata]
        for c in all_meta_cols:
            if df[c].nunique() > 1: # Only analyze if there's >1 group
                valid_meta_cols.append(c)
                
    # =========================================================================
    # Analysis 1: Overall Expression Differences Across Metadata
    # =========================================================================
    expression_diff_results = {}
    if valid_meta_cols:
        # Calculate cluster proportions per run
        run_cluster_counts = df.groupby(['run_id', 'cluster']).size().unstack(fill_value=0)
        run_cluster_props = run_cluster_counts.div(run_cluster_counts.sum(axis=1), axis=0)
        run_cluster_props.columns = [str(c) for c in run_cluster_props.columns]
        str_clusters = [str(c) for c in unique_clusters]
        
        # Ensure all columns exist for proportions
        for c in str_clusters:
            if c not in run_cluster_props.columns:
                run_cluster_props[c] = 0.0
                
        # Merge run proportions with metadata
        if metadata is not None:
            run_props_meta = run_cluster_props.reset_index().merge(metadata, on='run_id', how='left')
            
            for m_col in valid_meta_cols:
                # Mean and Std proportion per metadata group
                group_means = run_props_meta.groupby(m_col)[str_clusters].mean()
                group_stds = run_props_meta.groupby(m_col)[str_clusters].std().fillna(0)
                
                # Pairwise differences
                diffs = {}
                groups = group_means.index.tolist()
                for i in range(len(groups)):
                    for j in range(i+1, len(groups)):
                        g1, g2 = groups[i], groups[j]
                        diff = abs(group_means.loc[g1] - group_means.loc[g2]).to_dict()
                        diffs[f"{g1}_vs_{g2}"] = diff
                
                # NOUVEAU : Test statistique omnibus (Kruskal-Wallis)
                p_values_uncorrected = []
                test_stats = []
                clusters_tested = []
                
                for cluster_id in str_clusters:
                    samples = [run_props_meta[run_props_meta[m_col] == g][cluster_id].values for g in groups]
                    if len(samples) > 1 and all(len(s) > 0 for s in samples):
                        try:
                            stat, p_val = stats.kruskal(*samples)
                        except ValueError:
                            stat, p_val = np.nan, 1.0
                        test_stats.append(stat)
                        p_values_uncorrected.append(p_val)
                    else:
                        test_stats.append(np.nan)
                        p_values_uncorrected.append(1.0)
                    clusters_tested.append(cluster_id)
                
                # Correction FDR
                stats_results = {}
                if p_values_uncorrected:
                    p_values_clean = np.nan_to_num(p_values_uncorrected, nan=1.0)
                    reject, p_values_corrected, _, _ = multipletests(p_values_clean, alpha=0.05, method='fdr_bh')
                    for idx, c_id in enumerate(clusters_tested):
                        stats_results[c_id] = {
                            "kruskal_stat": float(test_stats[idx]) if not np.isnan(test_stats[idx]) else None,
                            "p_value_uncorrected": float(p_values_clean[idx]),
                            "p_value_fdr_corrected": float(p_values_corrected[idx]),
                            "significant_omnibus": bool(reject[idx])
                        }

                expression_diff_results[m_col] = {
                    "group_means": group_means.to_dict(),
                    "group_stds": group_stds.to_dict(),
                    "pairwise_differences": diffs,
                    "statistics": stats_results
                }
        
        with open(os.path.join(output_dir, "metadata_expression_differences.json"), "w") as f:
            json.dump(expression_diff_results, f, indent=4)

    # =========================================================================
    # Analysis 2: Temporal Expression Subplots
    # =========================================================================
    if valid_meta_cols:
        df['time_bin'] = df['token_idx'] // time_window_size
        
        for m_col in valid_meta_cols:
            temp_df = df.dropna(subset=[m_col])
            
            # Count cluster usage per run and time bin
            run_bin_counts = temp_df.groupby(['run_id', m_col, 'time_bin', 'cluster']).size().unstack(fill_value=0)
            run_bin_totals = run_bin_counts.sum(axis=1)
            # Find the proportion of each cluster for that particular run at that time bin
            run_bin_props = run_bin_counts.div(run_bin_totals, axis=0) 
            run_bin_props = run_bin_props.reset_index()
            
            fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 3 * n_clusters), sharex=True)
            if n_clusters == 1:
                axes = [axes]
            
            for c_idx, cluster in enumerate(unique_clusters):
                ax = axes[c_idx]
                if cluster not in run_bin_props.columns:
                    continue
                
                for g_val, g_data in run_bin_props.groupby(m_col):
                    # Aggregate across runs for this time bin and group
                    bin_stats = g_data.groupby('time_bin')[cluster].agg(['mean', 'std', 'count'])
                    
                    # Stop drawing if fewer than 2-3 runs remain to prevent weird spikes at the end
                    min_runs_required = max(2, int(0.1 * g_data['run_id'].nunique())) 
                    valid_bins = bin_stats[bin_stats['count'] >= min_runs_required]
                    
                    if valid_bins.empty:
                        continue
                        
                    mean_vals = valid_bins['mean']
                    sem_vals = (valid_bins['std'].fillna(0) / np.sqrt(valid_bins['count']))
                    
                    ax.plot(valid_bins.index, mean_vals, label=str(g_val), lw=2)
                    ax.fill_between(valid_bins.index, mean_vals - sem_vals, mean_vals + sem_vals, alpha=0.3)
                
                # Ajout de la p-value corrigée au titre
                pval_str = ""
                if m_col in expression_diff_results and "statistics" in expression_diff_results[m_col]:
                    if str(cluster) in expression_diff_results[m_col]["statistics"]:
                        pval = expression_diff_results[m_col]["statistics"][str(cluster)]["p_value_fdr_corrected"]
                        pval_str = f" (FDR p={pval:.3f})"

                ax.set_title(f"Cluster {cluster} Temporal Expression{pval_str}")
                ax.set_ylabel("Proportion")
                # Legend on the outside
                if c_idx == 0:
                    ax.legend(title=m_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
            axes[-1].set_xlabel(f"Time Bins (window size: {time_window_size})")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"temporal_expression_{m_col}.png"), bbox_inches="tight")
            plt.close()
        print(f"Saved temporal expression plots to {output_dir}")



    # =========================================================================
    # Analysis 3: Mean Kinematics per Cluster
    # =========================================================================
    kinematics_results = {}
    if kinematics is not None and len(kinematics) > 0:
        sample_run_id = list(kinematics.keys())[0]
        kin_features = [k for k in kinematics[sample_run_id].keys() if k != 'frame_idx']
        
        cluster_kin_collections = {c: {f: [] for f in kin_features} for c in unique_clusters}
        
        for run_id, group in df.groupby('run_id'):
            if run_id in kinematics:
                run_kin = kinematics[run_id]
                for _, row in group.iterrows():
                    tok = int(row['token_idx'])
                    c = int(row['cluster'])
                    for f in kin_features:
                        if tok < len(run_kin[f]):
                            cluster_kin_collections[c][f].append(run_kin[f][tok])
                            
        for c in unique_clusters:
            kinematics_results[int(c)] = {}
            for f in kin_features:
                vals = np.array(cluster_kin_collections[c][f], dtype=np.float64)
                valid_vals = vals[~np.isnan(vals)]
                
                if len(valid_vals) > 0:
                    kinematics_results[int(c)][f] = {
                        "mean": float(np.nanmean(valid_vals)),
                        "std": float(np.nanstd(valid_vals))
                    }
                else:
                    kinematics_results[int(c)][f] = {"mean": None, "std": None}
                    
        with open(os.path.join(output_dir, "mean_kinematics_per_cluster.json"), "w") as f:
            json.dump(kinematics_results, f, indent=4)
        
    # =========================================================================
    # Analysis 4: Cluster Transition Graphs
    # =========================================================================
    df = df.sort_values(['run_id', 'token_idx'])
    df['next_cluster'] = df.groupby('run_id')['cluster'].shift(-1)
    
    transitions = df.dropna(subset=['next_cluster']).copy()
    transitions['next_cluster'] = transitions['next_cluster'].astype(int)
    
    def compute_transition_matrix(trans_df):
        counts = pd.crosstab(trans_df['cluster'], trans_df['next_cluster'])
        counts = counts.reindex(index=unique_clusters, columns=unique_clusters, fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        return probs

    def plot_transition_graph(matrix, title, filename):
        G = nx.DiGraph()
        for i in unique_clusters:
            G.add_node(i)
            # ensure all edges computed even if tiny, filter out only the extremely small ones for drawing
            for j in unique_clusters:
                weight = matrix.loc[i, j]
                if weight > 0.02: # Keep > 2% probability to get a healthy graph
                    G.add_edge(i, j, weight=weight)
                    
        pos = nx.circular_layout(G)
        plt.figure(figsize=(9, 7))
        
        edges = G.edges(data=True)
        if edges:
            weights = [e[2]['weight'] for e in edges]
            edge_colors = [cm.plasma(w) for w in weights]
            edge_widths = [w * 6 for w in weights]
            
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightgray', edgecolors="black")
            nx.draw_networkx_labels(G, pos, font_weight='bold', font_size=12)
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, arrowstyle='-|>', 
                arrowsize=20, width=edge_widths, 
                edge_color=edge_colors, edge_cmap=cm.plasma, edge_vmin=0.0, edge_vmax=1.0,
                connectionstyle='arc3,rad=0.15'
            )
            
            # Draw colorbar
            sm = plt.cm.ScalarMappable(cmap=cm.plasma, norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            plt.colorbar(sm, ax=plt.gca(), label="Transition Probability", fraction=0.046, pad=0.04)
            
            # Add labels for larger edges
            edge_labels = {(u,v): f"{d['weight']:.2f}" for u,v,d in edges if d['weight'] > 0.15}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=9)
            
        plt.title(title, pad=20, fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=150)
        plt.close()
    print(f"Saved transition graphs to {output_dir}")


    global_matrix = compute_transition_matrix(transitions)
    plot_transition_graph(global_matrix, "Global Cluster Transitions", "transition_graph_global.png")
    
    transition_results = {"global": global_matrix.to_dict()}
    
    if valid_meta_cols:
        for m_col in valid_meta_cols:
            transition_results[m_col] = {}
            for g_val, g_data in transitions.groupby(m_col):
                mat = compute_transition_matrix(g_data)
                plot_transition_graph(mat, f"Transitions for {m_col}: {g_val}", 
                                      f"transition_graph_{m_col}_{g_val}.png")
                transition_results[m_col][str(g_val)] = mat.to_dict()
                
            diffs = {}
            groups = list(transitions[m_col].dropna().unique())
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    g1, g2 = groups[i], groups[j]
                    m1 = pd.DataFrame(transition_results[m_col][str(g1)])
                    m2 = pd.DataFrame(transition_results[m_col][str(g2)])
                    diff_mat = abs(m1 - m2)
                    diffs[f"{g1}_vs_{g2}"] = float(diff_mat.mean().mean())
            
            transition_results[m_col]["differences_mean_abs"] = diffs

    with open(os.path.join(output_dir, "cluster_transitions.json"), "w") as f:
        json.dump(transition_results, f, indent=4)
        
    
    return {
        "expression_differences": expression_diff_results,
        "kinematics": kinematics_results,
        "transitions": transition_results
    }
