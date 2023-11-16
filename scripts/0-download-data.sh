#!/bin/bash

# Activate the poetry shell and install dependencies
poetry shell && poetry install

# Function to download, unzip, and rename Littlesis data
download_littlesis_data() {
    local file="$1"
    wget -O "data/littlesis-$file.json.gz" "https://littlesis.org/public_data/$file.json.gz" && \
    gunzip "data/littlesis-$file.json.gz" && \
    mv "data/littlesis-$file.json" "data/littlesis-$file.json"
}

# Download and rename Littlesis entities and relationships data
for file in "entities" "relationships"; do
    download_littlesis_data "$file"
done

# Download and convert legislators-historical.yaml to JSON
wget -O data/legislators-historical.yaml https://github.com/unitedstates/congress-legislators/raw/main/legislators-historical.yaml && \
python scripts/yaml-to-json.py data/legislators-historical.yaml data/legislators-historical.json && \
rm data/legislators-historical.yaml
