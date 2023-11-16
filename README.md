![](DALL·E%20Banner.png)

# Donors And Ideologies

The code for my elective project, supervised by [Michele Coscia](https://www.michelecoscia.com/).

## How to use

[scripts/littlesis-graph-exploration.ipynb](scripts/littlesis-graph-exploration.ipynb) is a good place to start if you want to explore the data.

## Data

### Downloading the data

The only data included in the repository is the voteview data. This can also be downloaded from [voteview.com](https://voteview.com/) and save it as [data/voteview-HS117_members.csv](data/voteview-HS117_members.csv).

The rest of the data can be downloaded by running the following command:

```bash
sh scripts/download-data.sh
```

In case you want to download the data manually, you can find it here: [littlesis.org/data](https://littlesis.org/data)

Congress-legistlator data can be downloaded here: [github.com/unitedstates/congress-legislators](https://github.com/unitedstates/congress-legislators/blob/main/legislators-historical.yaml) and

### Data description

[data/littlesis-{entities,relationships}.json](data/) contain the graph nodes and edges. [littlesis](https://littlesis.org/) is a database of connections between people and organizations. I use the entities.json and the relationships.json files to create a graph of the connections between people and organizations.

[data/voteview-HS117_members.csv](data/voteview-HS117_members.csv) contains the ideology scores for the members of the 117th House of Representatives taken from [voteview.com](https://voteview.com/).

[data/congress-legislators](data/congress-legislators) contains data about the members of congress. It is taken from [github.com/unitedstates/congress-legislators](https://github.com/unitedstates/congress-legislators/blob/main/legislators-historical.yaml).

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

[file an issue]: https://github.com/DueViktor/donors-and-ideologies/issues
