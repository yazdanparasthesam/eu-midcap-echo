"""Kernel bootstrap: ensures the project root is on sys.path and required
packages are installed (called from notebooks as the very first cell)."""
import sys, subprocess, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def ensure(pkgs):
    """Try to import each package; pip-install any that are missing."""
    import importlib
    missing = []
    for mod, pip_name in pkgs.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"[bootstrap] installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# Mapping: import_name -> pip_name
DEPS = {
    "pandas_datareader": "pandas-datareader",
    "yfinance": "yfinance",
    "ta": "ta",
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "xgboost": "xgboost",
    "optuna": "optuna",
    "dotenv": "python-dotenv",
    "streamlit": "streamlit",
    "plotly": "plotly",
}
ensure(DEPS)
