"""
Everyone has their own way of naming things. Ids are no different. This script
will map all the ids to a common id. This will make it easier to merge the
dataframes later on. Littlesis, voteview, my own matrix id and more are included in this project.
"""
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from utils import read_json, save_json


def count_ids(entities: List[dict]):
    """
    Count how many entities have each of the IDs.
    """
    ids = []
    for entity in entities:
        for id in entity["ids"]:
            ids.append(id)
    return Counter(ids)


def build_entity_list(CHOSEN_CONGRESS):
    """
    Build a list of politicians for the given Congress and map various IDs to each other.
    """
    # Constants
    voteview_data = pd.read_csv(
        f"data/voteview-HS{CHOSEN_CONGRESS}_members.csv"
    ).to_dict(orient="records")
    entities = []

    for member in voteview_data:
        entity = {"voteview_info": member, "ids": {}}
        entities.append(entity)

    # Count how many entities have each of the IDs before mapping
    count_ids_before_mapping = count_ids(entities)

    # Map the ICPSR ID
    for entity in entities:
        icpsr = entity["voteview_info"]["icpsr"]
        entity["ids"]["icpsr"] = icpsr

    # Count how many entities have each of the IDs after mapping ICPSR
    count_ids_after_icpsr_mapping = count_ids(entities)

    # Map the Bioguide ID
    for entity in entities:
        bioguide_id = entity["voteview_info"]["bioguide_id"]
        if pd.isna(bioguide_id):
            bioguide_id = None
        entity["ids"]["bioguide_id"] = bioguide_id

    # Load legislators historical data
    legislators_historical = read_json("data/legislators-historical.json")

    # Map GovTrack ID from legislators historical data
    for legislator in legislators_historical:
        try:
            icpsr = legislator["id"]["icpsr"]
            govtrack = legislator["id"]["govtrack"]
        except KeyError:
            continue

        for entity in entities:
            if entity["ids"]["icpsr"] == icpsr:
                entity["ids"]["govtrack"] = govtrack
                entity["legislators-historical_info"] = legislator

    # Load LittleSis data
    littlesis_entities = read_json("data/littlesis-entities.json")

    # Map LittleSis ID based on GovTrack
    for littlesis_entity in littlesis_entities:
        try:
            elected_rep = littlesis_entity["attributes"]["extensions"][
                "ElectedRepresentative"
            ]
            govtrack_id = elected_rep["govtrack_id"]
        except KeyError:
            continue

        for entity in entities:
            try:
                govtrack_equal = str(entity["ids"]["govtrack"]) == govtrack_id
            except KeyError:
                govtrack_equal = False

            if govtrack_equal:
                entity["ids"]["littlesis"] = littlesis_entity["id"]
                entity["littlesis_info"] = littlesis_entity

    # Map LittleSis ID based on Bioguide
    for littlesis_entity in littlesis_entities:
        try:
            elected_rep = littlesis_entity["attributes"]["extensions"][
                "ElectedRepresentative"
            ]
            bioguide_id = elected_rep["bioguide_id"]
        except KeyError:
            continue

        for entity in entities:
            try:
                bioguide_equal = str(entity["ids"]["bioguide_id"]) == bioguide_id
            except KeyError:
                bioguide_equal = False

            if bioguide_equal or govtrack_equal:
                entity["ids"]["littlesis"] = littlesis_entity["id"]
                entity["littlesis_info"] = littlesis_entity

    # Count how many entities have each of the IDs after all mappings
    count_ids_after_all_mappings = count_ids(entities)

    # Save the entities to a JSON file
    save_json(data=entities, path=Path(f"data/politicians_{CHOSEN_CONGRESS}.json"))

    return (
        count_ids_before_mapping,
        count_ids_after_icpsr_mapping,
        count_ids_after_all_mappings,
    )


if __name__ == "__main__":
    CHOSEN_CONGRESS = 117
    count_before, count_after_icpsr, count_after_all = build_entity_list(
        CHOSEN_CONGRESS
    )
    print("Count of IDs before mapping:")
    print(count_before)
    print("Count of IDs after ICPSR mapping:")
    print(count_after_icpsr)
    print("Count of IDs after all mappings:")
    print(count_after_all)
