"""
Description:
    The script to build the matrix for node vector distance (NVD) calculations.
    The matrix is a matrix where each column is a politician and each row is a donor.
    The cell value is the amount of donation from the donor to the politician, log-transformed, 
    
    Perhaps TODO: and normalized by the total amount of donation from the donor (i.e., the sum of all cells in the row).

    The actual calculation is done in Julia, see scripts/5-nvd.jl.

    The script generates two files:
        network.csv - simply the edges of the graph in the format a,b
        node_attributes.csv - the matrix described above
"""

from pathlib import Path

import networkx as nx
import numpy as np
from tqdm import tqdm
from utils import read_json, save_json


def generate_custom_id(G_largest: nx.Graph) -> None:
    """
    Generate a custom ID for each node in G_largest.
    """
    littlesis_id_to_my_id = {}
    my_id_to_littlesis_id = {}

    for i, node_id in enumerate(G_largest.nodes()):
        my_id_to_littlesis_id[i + 1] = node_id
        littlesis_id_to_my_id[node_id] = i + 1

    # Save the mappings to JSON files
    save_json(
        data=littlesis_id_to_my_id, path=Path("data/id-mapping-littlesis-to-mine.json")
    )
    save_json(
        data=my_id_to_littlesis_id, path=Path("data/id-mapping-mine-to-littlesis.json")
    )


def generate_network_csv(G_largest: nx.Graph) -> None:
    """
    Generate network.csv, which contains the edges of the graph in the format "a,b."
    Note that the IDs are custom-created.
    """
    littlesis_id_to_my_id = read_json(
        path=Path("data/id-mapping-littlesis-to-mine.json")
    )

    with open("data/nvd-network.csv", "w") as f:
        for edge in tqdm(G_largest.edges(), desc="Writing network.csv"):
            littlesis_src_id, littlesis_dst_id = edge

            src_id = littlesis_id_to_my_id[str(littlesis_src_id)]
            dst_id = littlesis_id_to_my_id[str(littlesis_dst_id)]

            f.write(f"{src_id},{dst_id}\n")


def generate_node_attributes_csv(
    donation_to_politician: dict, my_id_to_littlesis_id: dict
) -> None:
    # Generate node_attributes.csv from donation_to_politician
    with open("data/nvd-node_attributes.csv", "w") as f:
        f.write("donor,")
        politician_ids = [
            str(politician_id) for politician_id in set(donation_to_politician.keys())
        ]
        f.write(",".join(politician_ids) + "\n")

        for my_donor_id, littlesis_donor_id in tqdm(
            my_id_to_littlesis_id.items(), desc="Writing nvd-node_attributes.csv"
        ):
            line = f"{my_donor_id},"
            littlesis_donor_id = str(littlesis_donor_id)
            for politician in politician_ids:
                if littlesis_donor_id in donation_to_politician[politician]:
                    donation = donation_to_politician[politician][littlesis_donor_id]
                    if donation < 0:  # Negative donations are not allowed
                        donation = 0
                    logged_donation = np.log(donation + 1)
                    line += f"{logged_donation},"
                else:
                    line += "0,"  # np.log(1) = 0
            f.write(line[:-1] + "\n")


if __name__ == "__main__":
    import pickle

    # Load the largest component in the Littlesis network
    G_largest = pickle.load(open("data/littlesis-largest-component.pickle", "rb"))

    # Generate custom IDs
    generate_custom_id(G_largest)

    # Generate network.csv
    generate_network_csv(G_largest)

    # Load donation data
    donation_to_politician = read_json(Path("data/donation_to_politician.json"))

    # Generate node attributes
    generate_node_attributes_csv(
        donation_to_politician,
        read_json(Path("data/id-mapping-mine-to-littlesis.json")),
    )
