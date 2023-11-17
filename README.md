![](DALL·E%20Banner.png)

# Donors And Ideologies

The code for my elective project, supervised by [Michele Coscia](https://www.michelecoscia.com/).

## How to use

- [Donors And Ideologies](#donors-and-ideologies)
  - [How to use](#how-to-use)
    - [Data](#data)
    - [Data description](#data-description)
    - [Initializing the database](#initializing-the-database)
    - [Node Vector Distance for ideology estimation](#node-vector-distance-for-ideology-estimation)
    - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Issues](#issues)

### Data

The only data included in the repository is the voteview data. This can also be downloaded from [voteview.com](https://voteview.com/) and save it as [data/sources/voteview-HS117_members.csv](data/sources/voteview-HS117_members.csv).

The rest of the data can be downloaded by running the following command:

```bash
sh scripts/download-data.sh
```

In case you want to download the data manually, you can find it here: [littlesis.org/data](https://littlesis.org/data)

Congress-legistlator data can be downloaded here: [github.com/unitedstates/congress-legislators](https://github.com/unitedstates/congress-legislators/blob/main/legislators-historical.yaml) and converted with [scripts/yaml-to-json.py](scripts/yaml-to-json.py)

### Data description

[data/sources/littlesis-{entities,relationships}.json](data/sources/) contain the graph nodes and edges. [littlesis](https://littlesis.org/) is a database of connections between people and organizations. I use the entities.json and the relationships.json files to create a graph of the connections between people and organizations.

[data/sources/voteview-HS117_members.csv](data/voteview-HS117_members.csv) contains the ideology scores for the members of the 117th House of Representatives taken from [voteview.com](https://voteview.com/).

[data/sources/legislators-historical.json](data/sources/legislators-historical.json) contains data about the members of congress. It is taken from [github.com/unitedstates/congress-legislators](https://github.com/unitedstates/congress-legislators/blob/main/legislators-historical.yaml).

### Initializing the database

Since I have data from many sources I decided to combine the information I need in a single "database". It is not a real database, but a `database.csv` with the neccesary information. The database is initialized by running the following command:

```bash
python scripts/initialize-database.py
```

### Node Vector Distance for ideology estimation

In order to perform the node vector distance experiment you need to run the following command:

```bash
python DonorIdeo/nvd.py
```

This code will break at a certain point because it needs the results from a julie script that is able to calculate distances between vectors much faster. This can be executed by running the following command:

```bash
julia scripts/nvd.jl
```

Note that it have a runningtime of approx 6 hours on a Macbook Air M1. When the julia script is done, you can run the python script `DonorIdeo/nvd.py` again and it will finish. It will add information to the `database.csv` file and create the following three plots in the `assets` folder.

![](assets/1d-projection.png)
![](assets/2d-projection.png)
![](assets/voteview-nominate-dim1.png)

### Exploratory Data Analysis

As seen in the section above, the node vector distance experiment requires a lot of time to run and does not give good results. The assumption was that the donations where a good indicator of ideology, but it seems that node vector calculations are not able to capture this. Therefore I decided to do some exploratory data analysis to see if I could find reasons in the data for this.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

[file an issue]: https://github.com/DueViktor/donors-and-ideologies/issues
