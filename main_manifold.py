import numpy as np
from cuml.manifold import TSNE, UMAP

def main():
    experiment = 'experiment3'  # Change this to experiment2 for the second set of embeddings
    DIMS = 50
    output_path = f"/scratch/izar/boesch/BehaveMAE/outputs/arena/{experiment}"
    dict_layer0_path = f"{output_path}/test_submission_0.npy"
    dict_layer1_path = f"{output_path}/test_submission_1.npy"
    dict_layer2_path = f"{output_path}/test_submission_2.npy"

    dict_layer0 = np.load(dict_layer0_path, allow_pickle=True).item()
    dict_layer1 = np.load(dict_layer1_path, allow_pickle=True).item()
    dict_layer2 = np.load(dict_layer2_path, allow_pickle=True).item()

    embeddings_layer0 = dict_layer0['embeddings'][:,:DIMS]
    embeddings_layer1 = dict_layer1['embeddings'][:,:DIMS]
    embeddings_layer2 = dict_layer2['embeddings'][:,:DIMS]

    tsne_layer0 = TSNE(n_components=2, random_state=0).fit_transform(embeddings_layer0)
    print("Done1")
    tsne_layer1 = TSNE(n_components=2, random_state=0).fit_transform(embeddings_layer1)
    print("Done2")

    tsne_layer2 = TSNE(n_components=2, random_state=0).fit_transform(embeddings_layer2)
    print("Done3")

    umap_layer0 = UMAP(n_components=2, random_state=0).fit_transform(embeddings_layer0)
    print("Done4")

    umap_layer1 = UMAP(n_components=2, random_state=0).fit_transform(embeddings_layer1)
    umap_layer2 = UMAP(n_components=2, random_state=0).fit_transform(embeddings_layer2)

    np.save(f"{output_path}/tsne_layer0.npy", tsne_layer0)
    np.save(f"{output_path}/tsne_layer1.npy", tsne_layer1)
    np.save(f"{output_path}/tsne_layer2.npy", tsne_layer2)

    np.save(f"{output_path}/umap_layer0.npy", umap_layer0)
    np.save(f"{output_path}/umap_layer1.npy", umap_layer1)
    np.save(f"{output_path}/umap_layer2.npy", umap_layer2)



if __name__ == "__main__":
    main()