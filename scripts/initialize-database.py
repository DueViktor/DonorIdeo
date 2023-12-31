"""
I have information from many sources (data/sources/), but I need to combine them into a single
file. This script does that.

Everyone has their own way of naming things. Ids are no different. This script
will map all the ids to a common id. This will make it easier to merge the
dataframes later on. Littlesis, voteview, my own matrix id and more are included in this project.
"""


import pandas as pd

from DonorIdeo.config import SOURCES_DATA_DIR
from DonorIdeo.json_utils import read_json


def initialize_with_voteview() -> pd.DataFrame:
    """Voteview provide the list of politicians"""
    voteview = pd.read_csv(SOURCES_DATA_DIR / "voteview-HS117_members.csv")

    columns_needed = [
        "congress",
        "chamber",
        "icpsr",
        "nominate_dim1",
        "bioname",
        "bioguide_id",
        "party_code",
    ]

    voteview = voteview[columns_needed]

    return voteview


def add_legislators_historical_info(db: pd.DataFrame) -> pd.DataFrame:
    # Map GovTrack ID from legislators historical data

    legislators_historical = read_json(
        SOURCES_DATA_DIR / "legislators-historical.json", verbose=True
    )

    for legislator in legislators_historical:
        try:
            icpsr: int = legislator["id"]["icpsr"]
            bioguide: str = legislator["id"]["bioguide"]
            govtrack: int = legislator["id"]["govtrack"]

            # Find the row in the db that has either the same icpsr or bioguide and add the govtrack id
            db.loc[db["icpsr"] == icpsr, "govtrack"] = govtrack
            db.loc[db["bioguide_id"] == bioguide, "govtrack"] = govtrack

        except KeyError:
            continue

    # how many politicians have a govtrack id?
    print(f"Number of politicians with a govtrack id: {db['govtrack'].count()}")

    return db


def add_littlesis_info(db: pd.DataFrame) -> pd.DataFrame:
    # load the littlesis entities
    littlesis_entities = read_json(
        SOURCES_DATA_DIR / "littlesis-entities.json", verbose=True
    )

    for entity in littlesis_entities:
        try:
            bioguide: str = entity["attributes"]["extensions"]["ElectedRepresentative"][
                "bioguide_id"
            ]

            if bioguide is None:
                continue

            assert isinstance(bioguide, str)

            # Find the row in the db that has the same bioguide and add the littlesis id
            db.loc[db["bioguide_id"] == bioguide, "littlesis"] = entity["id"]

        except KeyError:
            continue

    db["littlesis"] = db["littlesis"].astype("Int64")

    return db


def add_custom_info(db: pd.DataFrame) -> pd.DataFrame:
    def code_to_party(value):
        if value == 100:
            return "Democrat"
        elif value == 200:
            return "Republican"
        else:
            return "Other"

    # transform the party_code to a string
    db["party"] = db["party_code"].apply(code_to_party)

    # add plotting color for each party
    def party_to_color(value: str):
        if value == "Democrat":
            return "blue"
        elif value == "Republican":
            return "red"
        else:
            return "gray"

    db["color"] = db["party"].apply(party_to_color)

    # prepare for the projection
    db["projection-1d"] = [None for _ in range(len(db))]
    db["projection-2d_x"] = [None for _ in range(len(db))]
    db["projection-2d_y"] = [None for _ in range(len(db))]

    # donations
    db["donations-in-total"] = [0 for _ in range(len(db))]
    db["donations-from"] = ["<EMPTY>" for _ in range(len(db))]
    db["donations-count"] = [0 for _ in range(len(db))]

    return db


if __name__ == "__main__":
    from DonorIdeo.config import DATA_DIR

    # Note that the order matters here. Some ids can only be mapped after others have been mapped.
    db: pd.DataFrame = initialize_with_voteview()
    db: pd.DataFrame = add_littlesis_info(db)
    db: pd.DataFrame = add_custom_info(db)
    db.to_csv(DATA_DIR / "database.csv", index=False)
