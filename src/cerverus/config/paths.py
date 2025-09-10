from pathlib import Path

# Compute project root reliably by walking up from this file.
# src/cerverus/config/paths.py -> project root is parents[3]
ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = ROOT / "data" / "results"
EXTERNAL_DATA_DIR = ROOT / "data" / "external"


def ensure_dirs():
    """Create result and external data directories if they don't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
