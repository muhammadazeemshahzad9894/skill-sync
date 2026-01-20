# src/modules/project_embeddings/compute_similarity.py
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .store_similarity import save_similarity
from .store_embeddings import load_candidate_embeddings, load_project_embeddings


def compute_store_similarity():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_dir = os.path.join(root_dir, "data/artifacts")


    candidate_embeddings, candidate_ids = load_candidate_embeddings()


    project_embeddings, project_ids = load_project_embeddings()


    sim_matrix = cosine_similarity(candidate_embeddings, project_embeddings)


    
    save_similarity(sim_matrix, candidate_ids, project_ids)


    print("Candidate-project similarity matrix saved.")
    print(f"Matrix shape: {sim_matrix.shape} (candidates x projects)")




def compute_project_similarities(
    project_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str],
) -> dict[str, float]:
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_dir = os.path.join(root_dir, "data/artifacts")


    sims = cosine_similarity(
        project_embedding.reshape(1, -1),
        candidate_embeddings,
    )[0]
    return dict(zip(candidate_ids, sims))




