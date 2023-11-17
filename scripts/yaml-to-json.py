"""
Should be used like this:
    python scripts/yaml-to-json.py data/legislators-historical.yaml data/legislators-historical.json
"""

import json
import sys

import yaml

if len(sys.argv) != 3:
    print("Usage: python scripts/yaml-to-json.py <input-file> <output-file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

with open(output_file, "w") as f:
    json.dump(data, f)
