"""Utility helpers (paths, logging, config)."""
import os, logging, yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PRO = ROOT / "data" / "processed"
MODELS   = ROOT / "models"
REPORTS  = ROOT / "reports"
STORAGE  = ROOT / "storage"
SRC      = ROOT / "src"
for d in (DATA_RAW, DATA_PRO, MODELS, REPORTS, STORAGE):
    d.mkdir(parents=True, exist_ok=True)

UNIVERSE_PATH = SRC / "data" / "universe.yaml"

def load_universe():
    with open(UNIVERSE_PATH) as f:
        return yaml.safe_load(f)

def get_logger(name="eu-midcap-echo"):
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S")
    return logging.getLogger(name)

log = get_logger()
