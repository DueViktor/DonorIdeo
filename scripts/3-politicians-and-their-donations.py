"""
Description: 
    Collect all donations to politicians from the largest component of the LittleSis graph.
    
"""

import pickle
from collections import defaultdict

from tqdm import tqdm
from utils import read_json, save_json

politicians_117 = read_json("data/politicians_117.json")

with open("data/littlesis-largest-component.pickle", "rb") as file:
    G_largest = pickle.load(file)

donation_to_politician = {}
for entity in tqdm(G_largest.nodes()):
    id_ = G_largest.nodes[entity].get("id")
    if id_ is None:
        continue

    for politician in politicians_117:
        littlesis_id = politician.get("ids", {}).get("littlesis")
        if littlesis_id == id_:
            donation_to_politician[id_] = defaultdict(float)
            for src, dst, data in G_largest.edges(entity, data=True):
                if data.get("amount") is not None:
                    donation_to_politician[id_][dst] += data["amount"]

save_json(donation_to_politician, "data/donation_to_politician.json")
