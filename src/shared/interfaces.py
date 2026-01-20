
"""
Interface contracts for SkillSync modules
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CandidateProfile:
    """Candidate profile data structure"""
    id: str
    name: str
    skills: List[str]
    skill_levels: Dict[str, str]  # skill -> level (beginner, intermediate, advanced)
    role: str
    experience_years: int
    collaboration_style: str
    availability_hours: int
    tools: List[str]
    domains: List[str]
    
@dataclass
class ProjectDescription:
    """Project description data structure"""
    id: str
    title: str
    description: str
    required_roles: List[str]
    required_skills: List[str]
    team_size: int
    duration_weeks: int
    priority_skills: List[str]  # Must-have skills

# ============ Interface Functions ============

def extract_profiles_from_raw(raw_text: str) -> List[CandidateProfile]:
    """
    M5: Convert raw text profiles to structured CandidateProfile objects
    Returns: List of CandidateProfile objects
    """
    pass

def generate_candidate_embeddings(profiles: List[CandidateProfile]) -> Dict[str, Any]:
    """
    M1: Generate embeddings for all candidates and compute similarity matrix
    Returns: {
        'embeddings': Dict[str, np.array],  # candidate_id -> embedding vector
        'similarity_matrix': np.array,      # NxN similarity matrix
        'candidate_ids': List[str]          # Order of candidates in matrix
    }
    """
    pass

def generate_project_embedding(project: ProjectDescription) -> Dict[str, Any]:
    """
    M2: Generate embedding for project and compute candidate-project similarities
    Returns: {
        'project_embedding': np.array,
        'candidate_similarities': Dict[str, float],  # candidate_id -> similarity score
        'top_candidates': List[Tuple[str, float]]    # sorted list of (candidate_id, score)
    }
    """
    pass

def build_teams(
    profiles: List[CandidateProfile],
    project: ProjectDescription,
    candidate_sim_matrix: np.array,
    candidate_ids: List[str],
    project_similarities: Dict[str, float]
) -> List[List[CandidateProfile]]:
    """
    M3: Greedy algorithm for team formation
    Returns: List of teams (each team is list of CandidateProfile objects)
    """
    pass

def generate_explanations(
    teams: List[List[CandidateProfile]],
    project: ProjectDescription
) -> List[Dict[str, Any]]:
    """
    M5: Generate human-readable explanations for each team
    Returns: List of explanation objects for each team
    """
    pass
