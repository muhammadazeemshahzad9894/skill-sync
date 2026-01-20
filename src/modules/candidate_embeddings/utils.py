import json
from pathlib import Path

def load_profiles(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
