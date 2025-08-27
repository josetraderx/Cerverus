import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1] / 'fixtures'

def load_trades():
    p = ROOT / 'sample_trades.json'
    return json.loads(p.read_text())
