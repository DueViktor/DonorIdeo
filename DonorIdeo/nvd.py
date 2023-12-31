"""

See README.md for more information on this file.

The following code is used to compute a distance matrix and then perform
multidimensional scaling on the matrix to produce a 2D and 1D representation of the
data. 

The following methods for dimensionality have been tested:
- T-SNE (final choice)
- PCA
- Isomap
- UMAP

"""
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from tqdm import tqdm

from DonorIdeo.config import DATABASE_PATH, TEMPORARY_DATA_DIR
from DonorIdeo.json_utils import save_json
from DonorIdeo.littlesis_graph_utils import build_littlesis_graph, collect_donations_to


def generate_custom_id(G_largest: nx.Graph) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Generate a custom ID for each node"""
    littlesis_id_to_my_id = {}
    my_id_to_littlesis_id = {}

    for i, node_id in enumerate(G_largest.nodes()):
        my_id_to_littlesis_id[str(i + 1)] = str(node_id)
        littlesis_id_to_my_id[str(node_id)] = str(i + 1)

    # Save the mappings to JSON files
    save_json(
        data=littlesis_id_to_my_id,
        path=TEMPORARY_DATA_DIR / "id-mapping-littlesis-to-mine.json",
        verbose=True,
    )

    save_json(
        data=my_id_to_littlesis_id,
        path=TEMPORARY_DATA_DIR / "id-mapping-mine-to-littlesis.json",
        verbose=True,
    )

    return littlesis_id_to_my_id, my_id_to_littlesis_id


def generate_network_csv(
    littlesis_id_to_my_id: Dict[str, str], G_largest: nx.Graph
) -> None:
    """
    Generate network.csv, which contains the edges of the graph in the format "a,b."
    Note that the IDs are custom-created.
    """

    with open(TEMPORARY_DATA_DIR / "nvd-network.csv", "w") as f:
        for edge in tqdm(G_largest.edges(), desc="Writing network.csv"):
            littlesis_src_id, littlesis_dst_id = edge

            src_id = littlesis_id_to_my_id[str(littlesis_src_id)]
            dst_id = littlesis_id_to_my_id[str(littlesis_dst_id)]

            f.write(f"{src_id},{dst_id}\n")


def generate_node_attributes_csv(
    donation_to_politician: Dict[int, Dict[int, int]],
    my_id_to_littlesis_id: Dict[str, str],
) -> None:
    # Generate node_attributes.csv from donation_to_politician
    with open(TEMPORARY_DATA_DIR / "nvd-node_attributes.csv", "w") as f:
        # Write the header
        f.write("donor,")
        politician_ids: List[int] = [
            politician_id for politician_id in set(donation_to_politician.keys())
        ]
        f.write(",".join([str(polit) for polit in politician_ids]) + "\n")

        # Write the data
        for my_donor_id, littlesis_donor_id in tqdm(
            my_id_to_littlesis_id.items(), desc="Writing nvd-node_attributes.csv"
        ):
            # Write the donor ID
            line = f"{my_donor_id},"

            for politician in politician_ids:
                littlesis_donor_id: int = int(littlesis_donor_id)
                if littlesis_donor_id in donation_to_politician[politician]:
                    donation: int = donation_to_politician[politician][
                        littlesis_donor_id
                    ]

                    if donation < 0:  # Negative donations are not allowed
                        donation = 0
                    logged_donation: int = np.log(donation + 1)  # Add 1 to avoid log(0)
                    line += f"{logged_donation},"
                else:
                    line += "0,"  # np.log(1) = 0
            f.write(line[:-1] + "\n")  # Remove the last comma


def prepare_data_for_julia() -> None:
    """ """

    # 1. collect the donations to the politicians in the graph
    # Build the graph
    _, G_largest = build_littlesis_graph()

    # Collect the donations to the politicians in the graph
    littlesis_ids: List[int] = (
        pd.read_csv(DATABASE_PATH, dtype={"littlesis": "Int64"})["littlesis"]
        .dropna()
        .values.tolist()
    )
    donations_to_politician: Dict[int, Dict[int, int]] = collect_donations_to(
        politicians=littlesis_ids, graph=G_largest
    )

    save_json(
        data=donations_to_politician,
        path=TEMPORARY_DATA_DIR / "donations.json",
        verbose=True,
    )

    """ 2. Build the network.csv and node_attributes.csv
    
    Firstly I need to create a new_id for each of the nodes in the biggest connected components.
    This is because the julia script requires the nodes to be numbered from 1 to n.
    """
    littlesis_id_to_my_id, my_id_to_littlesis_id = generate_custom_id(G_largest)

    generate_network_csv(littlesis_id_to_my_id, G_largest)

    generate_node_attributes_csv(
        donation_to_politician=donations_to_politician,
        my_id_to_littlesis_id=my_id_to_littlesis_id,
    )


def compute_distance_matrix(write_to_file: bool = True) -> np.ndarray:
    node_attributes = pd.read_csv(TEMPORARY_DATA_DIR / "nvd-node_attributes.csv")

    politician_ids = node_attributes.columns[1:]
    politician_matrix_id = {
        int(polit_id): i for i, polit_id in enumerate(politician_ids)
    }
    distance_matrix = np.zeros((len(politician_ids), len(politician_ids)))
    for polit_id in tqdm(politician_ids, desc="Building distance matrix"):
        specific_path = f"data/nvd/{polit_id}.csv"
        distances = pd.read_csv(
            specific_path, skiprows=1, names=["src", "dst", "distance"]
        )
        # Setting all values
        for idx, row in distances.iterrows():
            src_matrix_id = int(politician_matrix_id[row["src"]])
            dst_matrix_id = int(politician_matrix_id[row["dst"]])
            dist = row["distance"]

            # set value in matrix
            distance_matrix[src_matrix_id, dst_matrix_id] = dist
            distance_matrix[dst_matrix_id, src_matrix_id] = dist

    if write_to_file:
        outpath: Path = TEMPORARY_DATA_DIR / "distance_matrix.csv"
        print(f"Writing distance matrix to file: {outpath}")
        np.savetxt(outpath, distance_matrix, delimiter=",")

    return distance_matrix


def project_and_scale(distance_matrix: np.ndarray, model: TSNE) -> np.ndarray:
    X_embedded = model.fit_transform(distance_matrix)

    # Scale the values to be between -1 and 1
    X_embedded = (X_embedded - X_embedded.min()) / (X_embedded.max() - X_embedded.min())
    X_embedded = (X_embedded * 2) - 1

    return X_embedded


def add_projections_to_database(
    projection_1d: np.ndarray,
    projection_2d: np.ndarray,
    politician_littlesis_ids: List[str],
    prefix: str = "",
) -> None:
    """IMPORTANT: I assume that the order of the politicians in the projection is the same"""
    database = pd.read_csv(DATABASE_PATH, dtype={"littlesis": "Int64"})

    # zip the politicians with their projections
    politicians_with_projections = zip(
        politician_littlesis_ids, projection_1d, projection_2d
    )

    # Add the projections to the database
    for littlesis_id, projection_1d, projection_2d in politicians_with_projections:
        database.loc[
            database["littlesis"] == int(littlesis_id), f"{prefix}projection-1d"
        ] = projection_1d
        database.loc[
            database["littlesis"] == int(littlesis_id), f"{prefix}projection-2d_x"
        ] = projection_2d[0]
        database.loc[
            database["littlesis"] == int(littlesis_id), f"{prefix}projection-2d_y"
        ] = projection_2d[1]

    # Save the database
    database.to_csv(DATABASE_PATH, index=False)


def reduced_vector_experiment():
    """This function is used to test if the distance matrix can be reduced to a smaller"""
    print("Running reduced_vector_experiment()")

    df = pd.read_csv(TEMPORARY_DATA_DIR / "nvd-node_attributes.csv", index_col=0)

    # How many columns sum to 0?
    cols_with_no_information = df.sum(axis=0) == 0
    cols_with_no_information = cols_with_no_information[
        cols_with_no_information
    ].index.values.tolist()

    # remove them
    print(f"\tRemoving columns with no information -> {cols_with_no_information}")
    df = df.drop(columns=cols_with_no_information)

    # How many rows sum to 0?
    rows_with_no_information = df.sum(axis=1) == 0
    rows_with_no_information = rows_with_no_information[
        rows_with_no_information
    ].index.values.tolist()

    # remove them
    tmp_size = df.shape[0]
    print(f"\tRemoving {len(rows_with_no_information)} rows with no information")
    df = df.drop(index=rows_with_no_information)
    print(f"\tNew size: {df.shape[0]}. Old size: {tmp_size}")
    exit()
    reduced_distance_matrix = np.zeros((df.shape[1], df.shape[1]))

    for i, col_1 in enumerate(df.columns):
        for j, col_2 in enumerate(df.columns):
            if i >= j:
                continue

            # consider col_1 and col_2 as vectors
            # calculate the euclidean distance between them

            euclidean_distance = np.linalg.norm(df[col_1] - df[col_2])

            reduced_distance_matrix[i, j] = euclidean_distance
            reduced_distance_matrix[j, i] = euclidean_distance

    # Project the distance matrix to 2D and scale the values to -1 to 1
    tsne_2 = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
    X_embedded_2 = project_and_scale(
        distance_matrix=reduced_distance_matrix, model=tsne_2
    )

    # Project the distance matrix to 1D and scale the values to -1 to 1
    tsne_1 = TSNE(n_components=1, metric="precomputed", init="random", random_state=42)
    X_embedded_1 = project_and_scale(
        distance_matrix=reduced_distance_matrix, model=tsne_1
    )

    # Update database.csv with the projection for each politician
    politician_ids: List[str] = df.columns.values.tolist()
    add_projections_to_database(
        projection_1d=X_embedded_1,
        projection_2d=X_embedded_2,
        politician_littlesis_ids=politician_ids,
        prefix="reduced-",
    )


if __name__ == "__main__":
    # import os

    # from DonorIdeo.config import NVD_DATA_DIR, TEMPORARY_DATA_DIR

    # # Compute network.csv and node_attributes that are needed for the julia script
    # prepare_data_for_julia()

    # # When the julia script has been run all results are in data/nvd
    # assert os.path.exists(
    #     TEMPORARY_DATA_DIR / "nvd-node_attributes.csv"
    # ), "The node_attributes.csv file is missing. Generated by the function prepare_data_for_julia()."
    # assert os.path.exists(
    #     TEMPORARY_DATA_DIR / "nvd-network.csv"
    # ), "The network.csv file is missing. Generated by the function prepare_data_for_julia()."

    # assert os.path.exists(
    #     NVD_DATA_DIR
    # ), "The nvd data directory is missing. Generated by the julia script."
    # assert (
    #     len(
    #         [
    #             politician_file
    #             for politician_file in os.listdir(NVD_DATA_DIR)
    #             if politician_file.endswith(".csv")
    #         ]
    #     )
    #     == 557
    # ), "The nvd data directory is missing some files. The Julia script are most likely not finished running."

    # # Compute the distance matrix
    # distance_matrix: np.ndarray = compute_distance_matrix(write_to_file=True)

    # # Project the distance matrix to 2D and scale the values to -1 to 1
    # tsne_2 = TSNE(n_components=2, metric="precomputed", init="random", random_state=42)
    # X_embedded_2 = project_and_scale(distance_matrix=distance_matrix, model=tsne_2)

    # # Project the distance matrix to 1D and scale the values to -1 to 1
    # tsne_1 = TSNE(n_components=1, metric="precomputed", init="random", random_state=42)
    # X_embedded_1 = project_and_scale(distance_matrix=distance_matrix, model=tsne_1)

    # # Update database.csv with the projection for each politician
    # politician_ids: List[str] = (
    #     pd.read_csv(TEMPORARY_DATA_DIR / "nvd-node_attributes.csv")
    #     .columns[1:]
    #     .values.tolist()
    # )

    # add_projections_to_database(
    #     projection_1d=X_embedded_1,
    #     projection_2d=X_embedded_2,
    #     politician_littlesis_ids=politician_ids,
    # )

    reduced_vector_experiment()
