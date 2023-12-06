from pathlib import Path

# PATHS

ROOT_DIR = Path(__file__).parent.parent

DATA_DIR = ROOT_DIR / "data"

DATABASE_PATH = DATA_DIR / "database.csv"
NVD_DATA_DIR = DATA_DIR / "nvd"
SOURCES_DATA_DIR = DATA_DIR / "sources"
TEMPORARY_DATA_DIR = DATA_DIR / "temporary"
ASSETS_DIR = ROOT_DIR / "assets"
MINIMUM_NVD_DIR = DATA_DIR / "minimum_nvd"
