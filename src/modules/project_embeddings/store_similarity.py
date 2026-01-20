import numpy as np
import json
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data/artifacts")


def save_similarity(sim_matrix, candidate_ids, project_ids,
                    npy_name="candidate_project_similarity.npy",
                    json_name="candidate_project_similarity.json"):
    np.save(os.path.join(DATA_DIR, npy_name), sim_matrix)
    payload = {
        "candidate_ids": candidate_ids,
        "project_ids": project_ids,
        "similarity_matrix": sim_matrix.tolist()
    }
    with open(os.path.join(DATA_DIR, json_name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)




def load_similarity(npy_name="candidate_project_similarity.npy",
                    json_name="candidate_project_similarity.json"):
    sim_matrix = np.load(os.path.join(DATA_DIR, npy_name))


    with open(os.path.join(DATA_DIR, json_name), "r", encoding="utf-8") as f:
        payload = json.load(f)


    return sim_matrix, payload["candidate_ids"], payload["project_ids"]





