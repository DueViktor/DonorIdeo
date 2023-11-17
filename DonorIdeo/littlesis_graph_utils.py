from typing import List, Tuple

import networkx as nx

from DonorIdeo.config import SOURCES_DATA_DIR
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


if __name__ == "__main__":
    import pandas as pd

    from DonorIdeo.config import DATABASE_PATH

    # Build the graph
    G, G_largest = build_littlesis_graph()

    # Load the politicians
    politicians: Tuple[str, int] = pd.read_csv(DATABASE_PATH)[
        ["bioname", "littlesis"]
    ].values.tolist()

    # Check whether the politicians are in the graph
    politicians_in_graph(politicians=politicians, graph=G_largest)
