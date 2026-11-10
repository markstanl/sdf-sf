import os
from pathlib import Path

PKG = Path(__file__).resolve().parent
DATA = Path(os.environ.get("REPRO_DATA", PKG / "data"))
OUT = Path(os.environ.get("REPRO_OUT", PKG / "out"))
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
