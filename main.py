
# """
# Main orchestrator for SkillSync system
# """

# import sys
# from pathlib import Path

# # Add src to path
# sys.path.append(str(Path(__file__).parent / 'src'))

# from modules.json_explainer import run_extraction


# from shared.interfaces import extract_profiles_from_raw, generate_candidate_embeddings, generate_project_embedding, build_teams, generate_explanations
# import json

# def main():
#     """
#     Main pipeline execution
#     """
#     print(' Starting SkillSync Pipeline')
    
#     # 1. Load data
#     # 2. Extract profiles
#     # 3. Generate embeddings
#     # 4. Build teams
#     # 5. Generate explanations
#     # 6. Output results
    
#     print('✅ Pipeline completed')

# if __name__ == '__main__':
#     main()


import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from modules.json_explainer import run_extraction

from shared.interfaces import (
    extract_profiles_from_raw,
    generate_candidate_embeddings,
    generate_project_embedding,
    build_teams,
    generate_explanations,
)

import json


def main():
    """
    Main pipeline execution
    """
    print(" Starting SkillSync Pipeline")

    print("DEBUG run_extraction imported from:", run_extraction.__module__)
    print("DEBUG run_extraction file:", run_extraction.__code__.co_filename)

  
    run_extraction(n=200)


    print("DEBUG finished run_extraction()")

    print(" Pipeline completed")


if __name__ == "__main__":
    main()
