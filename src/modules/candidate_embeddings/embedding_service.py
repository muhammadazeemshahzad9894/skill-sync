from pathlib import Path
from .utils import load_profiles
from .text_encoder import embed_profiles
from .similarity import cosine_dissimilarity_matrix
from .repository import (
    save_embeddings,
    save_dissimilarity
)

def gen_cand_emb_dissim(profiles_path: Path):
    
    profiles = load_profiles(profiles_path)
    embeddings = embed_profiles(profiles)
    candidate_ids = [p["id"] for p in profiles]

    save_embeddings(embeddings, candidate_ids)

    dissimilarity = cosine_dissimilarity_matrix(embeddings)
    save_dissimilarity(dissimilarity)

    print("Candidate Embeddings and dissimilarity matrix generated and saved successfully.")