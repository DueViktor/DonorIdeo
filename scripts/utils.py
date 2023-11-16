import json
from pathlib import Path
from typing import Dict


def read_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Dict, path: Path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
