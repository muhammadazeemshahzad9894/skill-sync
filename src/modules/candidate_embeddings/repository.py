import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = PROJECT_ROOT / "data" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)





def save_embeddings(embeddings, candidate_ids):
    np.save(ARTIFACT_DIR / "candidate_embeddings.npy", embeddings)
    with open(ARTIFACT_DIR / "candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)

def load_embeddings():
    embeddings = np.load(ARTIFACT_DIR / "candidate_embeddings.npy")
    with open(ARTIFACT_DIR / "candidate_ids.json") as f:
        candidate_ids = json.load(f)
    return embeddings, candidate_ids

def save_dissimilarity(matrix):
    np.save(ARTIFACT_DIR / "candidate_dissimilarity.npy", matrix)

def load_dissimilarity():
    return np.load(ARTIFACT_DIR / "candidate_dissimilarity.npy")
