import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_class_distribution(y, meta_var, output_dir):

    rotation = 90 if meta_var == "strain" else 45
    figsize = (12, 6) if meta_var == "strain" else (8, 5)
    plt.figure(figsize=figsize)
    sns.countplot(x=y, order=Counter(y).keys(), hue=y)

    plt.xlabel(meta_var)
    plt.ylabel("Count")
    plt.title(f"Class distribution for {meta_var}")
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{meta_var}_class_distribution.png"))
    plt.close()