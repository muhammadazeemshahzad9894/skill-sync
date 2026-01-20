import numpy as np
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")

def save_embeddings(embeddings, candidate_ids):
    np.save(os.path.join(DATA_DIR, "candidate_embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "candidate_ids.json"), "w") as f:
        json.dump(candidate_ids, f)

def load_embeddings():
    embeddings = np.load(os.path.join(DATA_DIR, "candidate_embeddings.npy"))
    with open(os.path.join(DATA_DIR, "candidate_ids.json"), "r") as f:
        candidate_ids = json.load(f)
    return embeddings, candidate_ids
