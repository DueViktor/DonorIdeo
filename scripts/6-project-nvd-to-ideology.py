"""
Description: This script takes the NVD data and projects it to the ideology space.
"""
import os
from typing import List

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from tqdm import tqdm
from utils import read_json, save_json

node_attributes = pd.read_csv("data/nvd-node_attributes.csv")

politician_ids = node_attributes.columns[1:]
politician_matrix_id = {int(polit_id): i for i, polit_id in enumerate(politician_ids)}
distance_matrix = np.zeros((len(politician_ids), len(politician_ids)))
for polit_id in tqdm(politician_ids, desc="Building distance matrix"):
    specific_path = f"data/nvd/{polit_id}.csv"
    distances = pd.read_csv(specific_path, skiprows=1, names=["src", "dst", "distance"])

    # Setting all values
    for idx, row in distances.iterrows():
        src_matrix_id = int(politician_matrix_id[row["src"]])
        dst_matrix_id = int(politician_matrix_id[row["dst"]])
        dist = row["distance"]

        # set value in matrix
        distance_matrix[src_matrix_id, dst_matrix_id] = dist
        distance_matrix[dst_matrix_id, src_matrix_id] = dist

# save the distance matrix
np.save("data/nvd-distance-matrix.npy", distance_matrix)


dimensionality_models = {
    "TSNE-2": TSNE(
        n_components=2, metric="precomputed", init="random", random_state=42
    ),
    "TSNE-1": TSNE(
        n_components=1, metric="precomputed", init="random", random_state=42
    ),
    "PCA-2": PCA(n_components=2, random_state=42),
    "PCA-1": PCA(n_components=1, random_state=42),
    "Isomap-2": Isomap(n_components=2, metric="precomputed"),
    "Isomap-1": Isomap(n_components=1, metric="precomputed"),
    "UMAP-2": umap.UMAP(n_components=2, metric="precomputed"),
    "UMAP-1": umap.UMAP(n_components=1, metric="precomputed"),
}

for name, model in dimensionality_models.items():
    print(f"Running {name}")
    X_embedded = model.fit_transform(distance_matrix)
    np.save(f"data/nvd-embeddings-{name}.npy", X_embedded)

    # Scale the values to be between -1 and 1
    X_embedded = (X_embedded - X_embedded.min()) / (X_embedded.max() - X_embedded.min())

    X_embedded = (X_embedded * 2) - 1

    # Save the embeddings
    np.save(f"data/nvd-embeddings-{name}.npy", X_embedded)

    # save embeddings to politician_117
    # Yes this is a stupid way to do this, but I'm tired and it works ;)
    politician_117 = read_json("data/politicians_117.json")
    nvds: List[str] = [f_.split(".")[0] for f_ in os.listdir("data/nvd")]
    updated_politician_117 = []
    for idx, politician_id in enumerate(politician_ids):
        # constant confusion about ids...
        assert str(politician_id) in nvds, f"{politician_id} not in nvds"

        for info_idx, info in enumerate(politician_117):
            try:
                littlesis_match: bool = int(info["ids"]["littlesis"]) == int(
                    politician_id
                )
            except KeyError:
                littlesis_match = False

            if littlesis_match:
                info[f"{name}"] = X_embedded[idx].tolist()

                politician_117[info_idx] = info

                break

    save_json(path="data/politicians_117.json", data=politician_117)
