import numpy as np
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/artifacts")

def save_project_embeddings(embeddings, project_ids):
    np.save(os.path.join(DATA_DIR, "project_embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "project_ids.json"), "w", encoding="utf-8") as f:
        json.dump(project_ids, f, indent=2)

def load_project_embeddings():
    embeddings = np.load(os.path.join(DATA_DIR, "project_embeddings.npy"))
    with open(os.path.join(DATA_DIR, "project_ids.json"), "r") as f:
        project_ids = json.load(f)
    return embeddings, project_ids

def save_candidate_embeddings(embeddings, candidate_ids):
    np.save(os.path.join(DATA_DIR, "candidate_embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "candidate_ids.json"), "w") as f:
        json.dump(candidate_ids, f)

def load_candidate_embeddings():
    embeddings = np.load(os.path.join(DATA_DIR, "candidate_embeddings.npy"))
    with open(os.path.join(DATA_DIR, "candidate_ids.json"), "r") as f:
        candidate_ids = json.load(f)
    return embeddings, candidate_ids

