import os
import numpy as np
import matplotlib.pyplot as plt


def plot_pca_explained_variance(pca, n_components=None, figsize=(14, 5), save_path=None):
    """
    Plot the explained variance by PCA components.
    
    Args:
        pca: Fitted sklearn PCA object with explained_variance_ratio_ attribute
        n_components: Number of components to plot. If None, plots all components.
        figsize: Figure size (width, height)
        save_path: Path to save the figure. If None, displays the plot.
    
    Returns:
        fig: Matplotlib figure object
    """
    if n_components is None:
        n_components = len(pca.explained_variance_ratio_)
    else:
        n_components = min(n_components, len(pca.explained_variance_ratio_))
    
    # Get explained variance for selected components
    explained_var = pca.explained_variance_ratio_[:n_components]
    cumsum_explained_var = np.cumsum(explained_var)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Individual explained variance ratio
    components = np.arange(1, n_components + 1)
    ax1.bar(components, explained_var, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('PCA Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Explained Variance by Component', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticks(components[::max(1, len(components)//10)])
    
    # Plot 2: Cumulative explained variance
    ax2.plot(components, cumsum_explained_var, marker='o', linestyle='-', 
             linewidth=2, markersize=6, color='darkgreen', label='Cumulative Explained Variance')
    ax2.fill_between(components, cumsum_explained_var, alpha=0.3, color='lightgreen')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(components[::max(1, len(components)//10)])
    
    # Add reference line for 95% variance
    ax2.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
    ax2.legend(loc='lower right', fontsize=10)
    
    # Add text showing total explained variance
    total_var = np.sum(explained_var)
    fig.suptitle(f'PCA Analysis ({n_components} components, Total Explained Variance: {total_var:.1%})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig
