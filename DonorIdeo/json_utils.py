import json
from pathlib import Path
from typing import Dict


def read_json(path: Path, verbose: bool) -> dict:
    if verbose:
        print(f"Reading {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: Path, verbose: bool) -> None:
    if verbose:
        print(f"Saving {path}")
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
