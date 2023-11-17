from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from DonorIdeo.config import DATABASE_PATH, SOURCES_DATA_DIR
from DonorIdeo.json_utils import read_json


def build_littlesis_graph() -> Tuple[nx.Graph, nx.Graph]:
    """Build a graph of the LittleSis data and return the fulle graph and the largest connected component"""

    G = nx.Graph()

    # Load the entities
    entities = read_json(SOURCES_DATA_DIR / "littlesis-entities.json", verbose=True)

    for entity in entities:
        G.add_node(
            entity["id"],
            id=entity["attributes"]["id"],
            name=entity["attributes"]["name"],
            primary_ext=entity["attributes"]["primary_ext"],
        )

    # Load the relationships
    relationships = read_json(
        SOURCES_DATA_DIR / "littlesis-relationships.json", verbose=True
    )
    for relationship in relationships:
        G.add_edge(
            relationship["attributes"]["entity1_id"],
            relationship["attributes"]["entity2_id"],
            category_id=relationship["attributes"]["category_id"],
            amount=relationship["attributes"]["amount"],
        )

    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_component: set = connected_components[0]
    G_largest = G.subgraph(largest_component)

    return G, G_largest


def politicians_in_graph(politicians: List[Tuple[str, int]], graph: nx.Graph):
    """Check whether the politicians are in the graph"""
    politicians_found = 0
    for name, littlesis_id in politicians:
        if littlesis_id not in graph.nodes:
            print(f"{name} not found in the biggest connected component.")
        else:
            politicians_found += 1

    print(
        f"Found {politicians_found}/{len(politicians)} politicians in the biggest connected component."
    )


def collect_donations_to(
    politicians: List[int], graph: nx.Graph, save_total_to_database: bool = True
) -> Dict[int, Dict[int, int]]:
    """Collect the donations to the politicians in the graph

    Returns:
        Dict: A dictionary with the politicians as keys and a dictionary of the donations as values and the keys as the donors
    """
    # fill the dictionary with the politicians
    donations_to_politician: Dict[int, Dict[int, int]] = {
        politician: {} for politician in politicians
    }

    for littlesis_id in politicians:
        for neighbor in graph.neighbors(littlesis_id):
            category_id: int = graph[littlesis_id][neighbor]["category_id"]
            donation_amount: int = graph[littlesis_id][neighbor]["amount"]

            # help to understand the category_id limitation: https://littlesis.org/help/relationships
            if category_id not in [5, 7] or donation_amount is None:
                continue

            # Get the donor
            donor_id = graph.nodes[neighbor]["id"]

            # Add the donation to the dictionary
            donations_to_politician[littlesis_id][donor_id] = donation_amount

    if save_total_to_database:
        print("Saving the total donations to the database...")
        database: pd.DataFrame = pd.read_csv(
            DATABASE_PATH, dtype={"littlesis": "Int64"}
        )
        for politician, donations in donations_to_politician.items():
            total_donations: int = sum(donations.values())
            # sum of donations to the politician
            database.loc[
                database["littlesis"] == politician, "donations-in-total"
            ] = total_donations
            # the keys of the donations
            donation_keys: str = ",".join([str(key) for key in donations.keys()])
            database.loc[
                database["littlesis"] == politician, "donations-from"
            ] = donation_keys
            # the number of donations
            database.loc[database["littlesis"] == politician, "donations-count"] = len(
                donations.keys()
            )
        database.to_csv(DATABASE_PATH, index=False)

    return donations_to_politician


if __name__ == "__main__":
    import pandas as pd

    from DonorIdeo.config import DATABASE_PATH, TEMPORARY_DATA_DIR
    from DonorIdeo.json_utils import save_json

    # Build the graph
    G, G_largest = build_littlesis_graph()

    # Check whether the politicians are in the graph
    politicians: List[Tuple[str, int]] = pd.read_csv(DATABASE_PATH)[
        ["bioname", "littlesis"]
    ].values.tolist()
    politicians_in_graph(politicians=politicians, graph=G_largest)

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
