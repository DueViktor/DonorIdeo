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


def politicians_in_graph(politicians: List, graph: nx.Graph):
    """Check whether the politicians are in the graph"""
    politicians_found = 0
    for politician in politicians:
        try:
            littlesis_id: int = politician["ids"]["littlesis"]
        except KeyError:
            print(
                f"Politician {politician['voteview_info']['bioname']} has no LittleSis ID."
            )
            continue

        if littlesis_id not in graph.nodes:
            print(
                f"Politician {politician['voteview_info']['bioname']} ({littlesis_id}) not found in the biggest connected component."
            )
        else:
            politicians_found += 1

    print(
        f"Found {politicians_found}/{len(politicians)} politicians in the biggest connected component."
    )


if __name__ == "__main__":
    # Build the graph
    G, G_largest = build_littlesis_graph()

    # Load the politicians
    politicians = read_json(SOURCES_DATA_DIR / "politicians_117.json", verbose=True)

    # Check whether the politicians are in the graph
    politicians_in_graph(politicians, G_largest)
