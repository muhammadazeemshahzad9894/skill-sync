import json


def load_projects(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
