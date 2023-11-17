#!/bin/bash

# Activate the poetry shell and install dependencies
poetry shell && poetry install

# Function to download, unzip, and rename Littlesis data
download_littlesis_data() {
    local file="$1"
    wget -O "data/sources/littlesis-$file.json.gz" "https://littlesis.org/public_data/$file.json.gz" && \
    gunzip "data/sources/littlesis-$file.json.gz" && \
    mv "data/sources/littlesis-$file.json" "data/sources/littlesis-$file.json"
}

# Download and rename Littlesis entities and relationships data
for file in "entities" "relationships"; do
    download_littlesis_data "$file"
done

# Download and convert legislators-historical.yaml to JSON
wget -O data/sources/legislators-historical.yaml https://github.com/unitedstates/congress-legislators/raw/main/legislators-historical.yaml && \
python scripts/yaml-to-json.py data/sources/legislators-historical.yaml data/sources/legislators-historical.json && \
rm data/sources/legislators-historical.yaml
