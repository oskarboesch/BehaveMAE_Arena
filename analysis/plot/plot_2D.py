import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_2D(x, label, dims=(0,1), is_discrete=True, title=None, figsize=(8,6), output_path=None, max_points=100000, random_state=42):
    if len(x) > max_points:
        np.random.seed(random_state)
        indices = np.random.choice(len(x), max_points, replace=False)
        x = x[indices]
        label = label[indices]
    plt.figure(figsize=figsize)
    # check if dims in x
    if dims[0] >= x.shape[1] or dims[1] >= x.shape[1]:
        print(f"Error: dims {dims} out of range for x with shape {x.shape}")
        return
    if is_discrete:
        sns.scatterplot(x=x[:, dims[0]], y=x[:, dims[1]], hue=label, palette="tab10", s=10, alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(x[:, dims[0]], x[:, dims[1]], c=label, cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(label=label.name if hasattr(label, "name") else "value")
    plt.xlabel(f"Dim {dims[0]}")
    plt.ylabel(f"Dim {dims[1]}")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    plt.close()