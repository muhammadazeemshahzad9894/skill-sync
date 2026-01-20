#!/bin/bash

# SkillSync Project Setup Script
# Run: chmod +x setup_skillsync.sh && ./setup_skillsync.sh

echo "========================================"
echo "🎯 SkillSync Project Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${2}${1}${NC}"
}

# Function to create directory if it doesn't exist
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_color "  Created: $1" "$GREEN"
    else
        print_color "  Exists: $1" "$YELLOW"
    fi
}

# Function to create file with content
create_file() {
    if [ ! -f "$1" ]; then
        printf '%s' "$2" > "$1"
        print_color "  Created: $1" "$GREEN"
    else
        print_color "  Exists: $1" "$YELLOW"
    fi
}

create_python_file() {
    local file_path="$1"
    local content="$2"
    
    if [ ! -f "$file_path" ]; then
        # Write content directly
        echo "$content" > "$file_path"
        print_color "  Created: $file_path" "$GREEN"
    else
        print_color "  Exists: $file_path" "$YELLOW"
    fi
}

print_color "📁 Creating project structure..." "$BLUE"

# # Create root directory
# create_dir "skill-sync"

# cd "skill-sync" || exit

# Create main directories
print_color "\n🏗️  Creating directory structure..." "$BLUE"
directories=(
    ".github/workflows"
    "docs"
    "src/modules/candidate_embeddings"
    "src/modules/project_embeddings"
    "src/modules/team_builder"
    "src/modules/ui"
    "src/modules/json_explainer"
    "src/shared"
    "tests"
    "data/sample"
    "outputs/embeddings"
    "outputs/teams"
    "outputs/explanations"
    "notebooks"
    "logs"
)

for dir in "${directories[@]}"; do
    create_dir "$dir"
done

print_color "\n📄 Creating configuration files..." "$BLUE"

# Create README.md
create_file "README.md" "# SkillSync - AI-Powered Team Formation

## 👥 Team 45
- Shahzad (M1): Candidate embeddings & similarity
- Roxana (M2): Project embeddings & candidate-project fit
- Ana (M3): Team construction algorithm
- Șaban (M4): Architecture & UI (Streamlit)
- Noor (M5): JSON extraction & team explanations

## 🚀 Quick Start
1. Install: \`pip install -r requirements.txt\`
2. Run UI: \`streamlit run src/modules/member4_ui/app.py\`
3. Test: \`python tests/integration_test.py\`

## 📁 Project Structure
See \`docs/architecture.md\` for detailed architecture.

## 🔧 Development
- Each member works in their \`src/modules/memberX_*\` directory
- Use \`shared/interfaces.py\` for module communication
- Create PRs for integration
"

# Create requirements.txt
create_file "requirements.txt" "streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
openai>=1.0.0
plotly>=5.17.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
faiss-cpu>=1.7.0
"

# Create .gitignore
create_file ".gitignore" "# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data
*.csv
*.pkl
*.pickle
*.h5
*.hdf5

# Logs
logs/
*.log

# Outputs
outputs/
"

# Create config.yaml
create_file "config.yaml" "# SkillSync Configuration

paths:
  data: \"./data\"
  outputs: \"./outputs\"
  models: \"./models\"
  logs: \"./logs\"

modules:
  json_extractor: \"src/modules/member5_json_explainer/extractor.py\"
  candidate_embeddings: \"src/modules/member1_candidate_embeddings/embeddings.py\"
  project_embeddings: \"src/modules/member2_project_embeddings/embeddings.py\"
  team_builder: \"src/modules/member3_team_builder/builder.py\"
  explanations: \"src/modules/member5_json_explainer/explanations.py\"
  ui: \"src/modules/member4_ui/app.py\"

settings:
  embedding_model: \"all-MiniLM-L6-v2\"
  embedding_dimension: 384
  similarity_threshold: 0.7
  max_team_size: 8
  min_team_size: 2

api_keys:
  openai: \${OPENAI_API_KEY}  # Load from .env file

logging:
  level: \"INFO\"
  format: \"%(asctime)s - %(name)s - %(levelname)s - %(message)s\"
"

# Create .env template
create_file ".env.template" "# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Application Settings
DEBUG=True
LOG_LEVEL=INFO

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
"

print_color "\n🐍 Creating Python modules..." "$BLUE"

# Create shared interfaces
create_file "src/shared/interfaces.py" """
\"\"\"
Interface contracts for SkillSync modules
\"\"\"

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CandidateProfile:
    \"\"\"Candidate profile data structure\"\"\"
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
    \"\"\"Project description data structure\"\"\"
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
    \"\"\"
    M5: Convert raw text profiles to structured CandidateProfile objects
    Returns: List of CandidateProfile objects
    \"\"\"
    pass

def generate_candidate_embeddings(profiles: List[CandidateProfile]) -> Dict[str, Any]:
    \"\"\"
    M1: Generate embeddings for all candidates and compute similarity matrix
    Returns: {
        'embeddings': Dict[str, np.array],  # candidate_id -> embedding vector
        'similarity_matrix': np.array,      # NxN similarity matrix
        'candidate_ids': List[str]          # Order of candidates in matrix
    }
    \"\"\"
    pass

def generate_project_embedding(project: ProjectDescription) -> Dict[str, Any]:
    \"\"\"
    M2: Generate embedding for project and compute candidate-project similarities
    Returns: {
        'project_embedding': np.array,
        'candidate_similarities': Dict[str, float],  # candidate_id -> similarity score
        'top_candidates': List[Tuple[str, float]]    # sorted list of (candidate_id, score)
    }
    \"\"\"
    pass

def build_teams(
    profiles: List[CandidateProfile],
    project: ProjectDescription,
    candidate_sim_matrix: np.array,
    candidate_ids: List[str],
    project_similarities: Dict[str, float]
) -> List[List[CandidateProfile]]:
    \"\"\"
    M3: Greedy algorithm for team formation
    Returns: List of teams (each team is list of CandidateProfile objects)
    \"\"\"
    pass

def generate_explanations(
    teams: List[List[CandidateProfile]],
    project: ProjectDescription
) -> List[Dict[str, Any]]:
    \"\"\"
    M5: Generate human-readable explanations for each team
    Returns: List of explanation objects for each team
    \"\"\"
    pass
"""

# Create M4 UI app
create_file "src/modules/ui/app.py" """
import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.interfaces import CandidateProfile, ProjectDescription

# Page configuration
st.set_page_config(
    page_title=\"SkillSync - AI Team Formation\",
    page_icon=\"🎯\",
    layout=\"wide\",
    initial_sidebar_state=\"expanded\"
)

# Custom CSS
st.markdown(\"\"\"
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .team-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border-left: 5px solid #3B82F6;
    }
    .skill-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
\"\"\", unsafe_allow_html=True)

# Initialize session state
if 'profiles' not in st.session_state:
    st.session_state.profiles = []
if 'project' not in st.session_state:
    st.session_state.project = None
if 'teams' not in st.session_state:
    st.session_state.teams = []
if 'explanations' not in st.session_state:
    st.session_state.explanations = []

# Title
st.markdown(\"<h1 class='main-header'>🎯 SkillSync</h1>\", unsafe_allow_html=True)
st.markdown(\"### AI-Powered Team Formation System\")

# Sidebar navigation
with st.sidebar:
    st.image(\"https://img.icons8.com/color/96/000000/teamwork.png\", width=80)
    st.markdown(\"### Navigation\")
    
    page = st.radio(
        \"Go to\",
        [\"📤 Upload Data\", \"⚙️ Project Setup\", \"👥 Build Teams\", \"📊 Analytics\", \"📋 Results\"],
        label_visibility=\"collapsed\"
    )
    
    st.markdown(\"---\")
    st.markdown(\"### Team Status\")
    
    # Display module status
    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric(\"Profiles\", len(st.session_state.profiles) if st.session_state.profiles else \"0\")
    with status_cols[1]:
        st.metric(\"Teams\", len(st.session_state.teams))
    
    st.markdown(\"---\")
    st.markdown(\"**Team Members:**\")
    st.markdown(\"\"\"
    - Shahzad (M1): Candidate Embeddings
    - Roxana (M2): Project Embeddings  
    - Ana (M3): Team Algorithm
    - Noor (M5): JSON & Explanations
    \"\"\")

# Page 1: Upload Data
if page == \"📤 Upload Data\":
    st.header(\"📤 Upload Candidate Profiles\")
    
    tab1, tab2 = st.tabs([\"Upload JSON\", \"Raw Text Input\"])
    
    with tab1:
        st.markdown(\"Upload structured JSON profiles (from M5's extraction)\")
        uploaded_file = st.file_uploader(\"Choose JSON file\", type=['json'])
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                # Convert to CandidateProfile objects
                # This is a placeholder - will be replaced by M5's actual extraction
                st.session_state.profiles = data
                st.success(f\"✅ Successfully loaded {len(data)} profiles\")
                
                # Preview
                with st.expander(\"Preview Profiles\"):
                    df = pd.DataFrame(data[:5])
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f\"Error loading file: {e}\")
    
    with tab2:
        st.markdown(\"Paste raw profile text for M5 to process:\")
        raw_text = st.text_area(\"Raw profiles (one per line):\", height=200,
                               placeholder=\"John Doe\\nSkills: Python, React, SQL\\nRole: Full Stack Developer\\n...\")
        
        if st.button(\"Extract to JSON\"):
            if raw_text:
                st.info(\"🔄 This will call M5's JSON extraction module\")
                # Placeholder for M5's function
                # extracted = extract_profiles_from_raw(raw_text)
                st.session_state.profiles = [{\"name\": \"Sample\", \"skills\": [\"Python\"]}]  # Placeholder
                st.success(\"Profiles extracted (placeholder)\")
            else:
                st.warning(\"Please enter some profile text\")

# Page 2: Project Setup
elif page == \"⚙️ Project Setup\":
    st.header(\"⚙️ Project Requirements\")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_title = st.text_input(\"Project Title\", placeholder=\"E-commerce Platform Redesign\")
        project_desc = st.text_area(\"Project Description\", height=150,
                                   placeholder=\"We need to redesign our e-commerce platform with new features...\")
    
    with col2:
        team_size = st.slider(\"Team Size\", 2, 10, 4)
        duration = st.selectbox(\"Duration\", [\"2 weeks\", \"1 month\", \"3 months\", \"6 months+\"])
    
    st.markdown(\"### Required Roles & Skills\")
    
    role_col, skill_col = st.columns(2)
    
    with role_col:
        required_roles = st.multiselect(
            \"Required Roles\",
            [\"Frontend Developer\", \"Backend Developer\", \"UX Designer\", \"Data Scientist\", 
             \"Project Manager\", \"QA Engineer\", \"DevOps\", \"ML Engineer\"],
            default=[\"Frontend Developer\", \"Backend Developer\"]
        )
    
    with skill_col:
        required_skills = st.multiselect(
            \"Required Skills\",
            [\"Python\", \"JavaScript\", \"React\", \"Node.js\", \"Figma\", \"SQL\", \"Docker\", 
             \"AWS\", \"ML\", \"Data Analysis\", \"UI/UX\", \"Agile\"],
            default=[\"Python\", \"React\", \"SQL\"]
        )
    
    if st.button(\"🚀 Generate Team Recommendations\", type=\"primary\"):
        if not project_desc:
            st.warning(\"Please enter a project description\")
        else:
            # Create project object
            project = {
                \"title\": project_title,
                \"description\": project_desc,
                \"required_roles\": required_roles,
                \"required_skills\": required_skills,
                \"team_size\": team_size,
                \"duration\": duration
            }
            st.session_state.project = project
            
            # Show processing steps
            with st.status(\"Building optimal teams...\", expanded=True) as status:
                st.write(\"📊 Step 1: Generating candidate embeddings (M1)\")
                st.write(\"🎯 Step 2: Generating project embedding (M2)\")
                st.write(\"👥 Step 3: Running team construction algorithm (M3)\")
                st.write(\"💬 Step 4: Generating explanations (M5)\")
                
                # Simulate processing
                import time
                time.sleep(2)
                
                # Placeholder teams
                st.session_state.teams = [
                    [{\"name\": \"Alice\", \"role\": \"Frontend\", \"skills\": [\"React\", \"JavaScript\"]},
                     {\"name\": \"Bob\", \"role\": \"Backend\", \"skills\": [\"Python\", \"Node.js\"]}],
                    [{\"name\": \"Charlie\", \"role\": \"Full Stack\", \"skills\": [\"React\", \"Python\"]},
                     {\"name\": \"Diana\", \"role\": \"UX\", \"skills\": [\"Figma\", \"UI/UX\"]}]
                ]
                
                st.session_state.explanations = [
                    \"Team 1 has strong frontend-backend separation with complementary skills.\",
                    \"Team 2 features versatile full-stack developers with design expertise.\"
                ]
                
                status.update(label=\"✅ Teams generated!\", state=\"complete\")
            
            st.success(f\"Generated {len(st.session_state.teams)} team configurations\")
            st.rerun()

# Page 3: Build Teams (Algorithm Visualization)
elif page == \"👥 Build Teams\":
    st.header(\"👥 Team Construction Process\")
    
    if not st.session_state.project:
        st.warning(\"Please set up a project first on the Project Setup page\")
        st.stop()
    
    st.markdown(\"### Algorithm Visualization\")
    
    # Placeholder for M3's algorithm visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(\"Candidate-Project Fit\", \"0.82\", \"+0.12 vs random\")
    
    with col2:
        st.metric(\"Team Diversity\", \"0.76\", \"+0.18 vs random\")
    
    with col3:
        st.metric(\"Skill Coverage\", \"92%\", \"+24% vs random\")
    
    # Algorithm steps visualization
    st.markdown(\"#### Greedy Algorithm Steps\")
    
    steps = [
        \"1. Start with candidate with highest project fit\",
        \"2. Add candidate that maximizes: α*fit + β*diversity + γ*coverage\",
        \"3. Repeat until team size reached\",
        \"4. Output optimized team\"
    ]
    
    for step in steps:
        st.markdown(f\"<div style='padding: 10px; margin: 5px 0; background: #f0f9ff; border-radius: 5px;'>{step}</div>\", 
                   unsafe_allow_html=True)
    
    # Parameters control
    st.markdown(\"#### Algorithm Parameters\")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        alpha = st.slider(\"α: Fit Weight\", 0.0, 1.0, 0.5, 0.1)
    
    with param_col2:
        beta = st.slider(\"β: Diversity Weight\", 0.0, 1.0, 0.3, 0.1)
    
    with param_col3:
        gamma = st.slider(\"γ: Coverage Weight\", 0.0, 1.0, 0.2, 0.1)

# Page 4: Analytics
elif page == \"📊 Analytics\":
    st.header(\"📊 Analytics & Insights\")
    
    if not st.session_state.profiles:
        st.warning(\"Please upload profiles first\")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs([\"Similarity Matrix\", \"Skill Distribution\", \"Team Analytics\"])
    
    with tab1:
        st.markdown(\"### Candidate-Candidate Similarity (M1)\")
        # Placeholder for M1's similarity matrix
        st.info(\"This will display the similarity matrix from M1's module\")
        
        # Create sample matrix
        np.random.seed(42)
        n_profiles = min(10, len(st.session_state.profiles))
        sample_matrix = np.random.rand(n_profiles, n_profiles)
        np.fill_diagonal(sample_matrix, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=sample_matrix,
            colorscale='Blues',
            zmin=0, zmax=1
        ))
        
        fig.update_layout(
            title=\"Similarity Matrix (Sample)\",
            xaxis_title=\"Candidate Index\",
            yaxis_title=\"Candidate Index\",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown(\"### Skill Distribution\")
        # Placeholder for skill distribution
        skills_data = {\"Python\": 15, \"JavaScript\": 12, \"React\": 10, \"SQL\": 8, \"Figma\": 6, \"Docker\": 5}
        
        fig = px.bar(
            x=list(skills_data.keys()),
            y=list(skills_data.values()),
            title=\"Skill Frequency in Profiles\",
            labels={\"x\": \"Skill\", \"y\": \"Count\"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown(\"### Team Composition Analysis\")
        # Placeholder for team analytics
        st.info(\"Team analytics will appear here after team generation\")

# Page 5: Results
elif page == \"📋 Results\":
    st.header(\"📋 Team Recommendations\")
    
    if not st.session_state.teams:
        st.warning(\"No teams generated yet. Please generate teams on the Project Setup page.\")
        st.stop()
    
    st.markdown(f\"### Project: **{st.session_state.project.get('title', 'Untitled Project')}**\")
    st.markdown(f\"*{st.session_state.project.get('description', '')}*\")
    
    # Team selection
    selected_team = st.selectbox(
        \"Select Team to View\",
        [f\"Team {i+1}\" for i in range(len(st.session_state.teams))],
        index=0
    )
    
    team_idx = int(selected_team.split(\" \")[1]) - 1
    
    if team_idx < len(st.session_state.teams):
        team = st.session_state.teams[team_idx]
        explanation = st.session_state.explanations[team_idx] if team_idx < len(st.session_state.explanations) else \"\"
        
        # Team card
        st.markdown(\"<div class='team-card'>\", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(\"#### 👥 Team Members\")
            for member in team:
                st.markdown(f\"**{member.get('name', 'Unknown')}** - {member.get('role', 'No role')}\")
                
                # Skills badges
                skills = member.get('skills', [])
                if skills:
                    skill_html = \" \".join([f\"<span class='skill-badge'>{skill}</span>\" for skill in skills[:5]])
                    st.markdown(skill_html, unsafe_allow_html=True)
                st.markdown(\"---\")
        
        with col2:
            st.markdown(\"#### 📈 Metrics\")
            st.metric(\"Project Fit\", \"0.85\")
            st.metric(\"Diversity\", \"0.78\")
            st.metric(\"Skill Coverage\", \"95%\")
        
        st.markdown(\"</div>\", unsafe_allow_html=True)
        
        # Explanation
        st.markdown(\"#### 💬 AI Explanation\")
        st.info(explanation if explanation else \"Explanation will be generated by M5's module\")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(\"✅ Select This Team\", type=\"primary\"):
                st.success(\"Team selected! 🎉\")
        
        with col2:
            if st.button(\"🔄 Regenerate Team\"):
                st.info(\"Regenerating team...\")
        
        with col3:
            if st.button(\"📥 Export as JSON\"):
                # Export functionality
                import io
                buffer = io.StringIO()
                json.dump(team, buffer, indent=2)
                st.download_button(
                    label=\"Download JSON\",
                    data=buffer.getvalue(),
                    file_name=f\"team_{team_idx+1}.json\",
                    mime=\"application/json\"
                )
    
    # All teams overview
    st.markdown(\"---\")
    st.markdown(\"### All Generated Teams\")
    
    for i, team in enumerate(st.session_state.teams):
        with st.expander(f\"Team {i+1} - Click to expand\"):
            cols = st.columns(min(4, len(team)))
            for idx, member in enumerate(team):
                col_idx = idx % 4
                with cols[col_idx]:
                    st.markdown(f\"**{member.get('name', 'Member')}**\")
                    st.caption(member.get('role', 'Role'))
                    
                    skills = member.get('skills', [])
                    if skills:
                        st.write(\", \".join(skills[:3]))

# Footer
st.markdown(\"---\")
st.markdown(
    \"<div style='text-align: center; color: #666; font-size: 0.9rem;'>\"
    \"SkillSync Team Formation System • Group 45 • TU Wien • 2025</div>\",
    unsafe_allow_html=True
)
"""

print_color "\n🧪 Creating test files..." "$BLUE"

# Create integration test
create_file "tests/integration_test.py" """#!/usr/bin/env python3
\"""
Integration test for SkillSync system
Run with: python tests/integration_test.py
\"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_integration():
    \"\"\"Test the complete SkillSync pipeline\"\"\"
    print(\"🚀 Starting SkillSync Integration Test\")
    print(\"=\" * 50)
    
    # Step 1: Load sample data
    print(\"📥 Step 1: Loading sample data...\")
    try:
        with open('data/sample/sample_profiles.json') as f:
            profiles = json.load(f)
        print(f\"   ✅ Loaded {len(profiles)} sample profiles\")
    except FileNotFoundError:
        print(\"   ⚠️  Sample profiles not found, using mock data\")
        profiles = [
            {
                \"id\": \"1\",
                \"name\": \"Alice\",
                \"skills\": [\"Python\", \"ML\", \"SQL\"],
                \"role\": \"Data Scientist\",
                \"experience_years\": 3
            },
            {
                \"id\": \"2\", 
                \"name\": \"Bob\",
                \"skills\": [\"React\", \"JavaScript\", \"UI/UX\"],
                \"role\": \"Frontend Developer\",
                \"experience_years\": 2
            }
        ]
    
    # Step 2: M5 - JSON extraction (placeholder)
    print(\"\\n📝 Step 2: JSON Extraction (M5)\")
    print(\"   ✅ Placeholder: Would extract structured profiles from raw text\")
    
    # Step 3: M1 - Candidate embeddings
    print(\"\\n🔤 Step 3: Candidate Embeddings (M1)\")
    print(\"   ✅ Placeholder: Would generate embeddings and similarity matrix\")
    
    # Step 4: M2 - Project embedding
    print(\"\\n🎯 Step 4: Project Embedding (M2)\")
    print(\"   ✅ Placeholder: Would generate project embedding and fit scores\")
    
    # Step 5: M3 - Team construction
    print(\"\\n👥 Step 5: Team Construction (M3)\")
    print(\"   ✅ Placeholder: Would run greedy algorithm to form teams\")
    
    # Step 6: M5 - Explanations
    print(\"\\n💬 Step 6: Team Explanations (M5)\")
    print(\"   ✅ Placeholder: Would generate human-readable explanations\")
    
    # Step 7: M4 - UI display
    print(\"\\n🖥️  Step 7: UI Integration (M4)\")
    print(\"   ✅ Placeholder: Would display results in Streamlit app\")
    
    print(\"\\n\" + \"=\" * 50)
    print(\"🎉 Integration test completed successfully!\")
    print(\"\\nNext steps:\")
    print(\"1. Each member implements their module\")
    print(\"2. Replace placeholders with actual implementations\")
    print(\"3. Run: streamlit run src/modules/member4_ui/app.py\")

if __name__ == \"__main__\":
    test_integration()
"""

# Create sample data
create_file "data/sample/sample_profiles_raw.txt" """John Doe
Senior Backend Developer with 5 years experience
Skills: Python (Expert), Django (Advanced), PostgreSQL (Advanced), Docker (Intermediate), AWS (Intermediate)
Tools: Git, Docker, Jenkins, Kubernetes
Collaboration Style: Prefers agile methodology, good at mentoring juniors
Availability: 40 hours/week
Timezone: UTC+1

Jane Smith
UX/UI Designer with 3 years experience  
Skills: Figma (Expert), Adobe XD (Advanced), User Research (Advanced), Prototyping (Advanced)
Tools: Sketch, InVision, Zeplin, Miro
Collaboration Style: Creative, enjoys brainstorming sessions, good communicator
Availability: 35 hours/week
Timezone: UTC+2

Alex Chen
Data Scientist with 4 years experience
Skills: Python (Advanced), Machine Learning (Advanced), SQL (Advanced), TensorFlow (Intermediate), PyTorch (Intermediate)
Tools: Jupyter, Scikit-learn, Pandas, NumPy
Collaboration Style: Analytical, detail-oriented, prefers structured tasks
Availability: 40 hours/week  
Timezone: UTC+1

Maria Garcia
Full Stack Developer with 6 years experience
Skills: JavaScript (Expert), React (Expert), Node.js (Advanced), MongoDB (Advanced), AWS (Intermediate)
Tools: VS Code, Git, Docker, Jest
Collaboration Style: Team player, proactive, good at code reviews
Availability: 30 hours/week
Timezone: UTC-5

David Kim
Project Manager with 7 years experience
Skills: Agile (Expert), Scrum (Expert), Risk Management (Advanced), Budgeting (Advanced)
Tools: Jira, Confluence, Trello, Slack
Collaboration Style: Leadership, organized, good at conflict resolution
Availability: 40 hours/week
Timezone: UTC+0
"""

create_file "data/sample/sample_profiles.json" """[
  {
    \"id\": \"1\",
    \"name\": \"John Doe\",
    \"role\": \"Backend Developer\",
    \"skills\": [\"Python\", \"Django\", \"PostgreSQL\", \"Docker\", \"AWS\"],
    \"skill_levels\": {
      \"Python\": \"Expert\",
      \"Django\": \"Advanced\", 
      \"PostgreSQL\": \"Advanced\",
      \"Docker\": \"Intermediate\",
      \"AWS\": \"Intermediate\"
    },
    \"experience_years\": 5,
    \"tools\": [\"Git\", \"Docker\", \"Jenkins\", \"Kubernetes\"],
    \"collaboration_style\": \"Agile, mentoring\",
    \"availability_hours\": 40,
    \"timezone\": \"UTC+1\"
  },
  {
    \"id\": \"2\",
    \"name\": \"Jane Smith\",
    \"role\": \"UX/UI Designer\", 
    \"skills\": [\"Figma\", \"Adobe XD\", \"User Research\", \"Prototyping\", \"UI/UX Design\"],
    \"skill_levels\": {
      \"Figma\": \"Expert\",
      \"Adobe XD\": \"Advanced\",
      \"User Research\": \"Advanced\",
      \"Prototyping\": \"Advanced\"
    },
    \"experience_years\": 3,
    \"tools\": [\"Sketch\", \"InVision\", \"Zeplin\", \"Miro\"],
    \"collaboration_style\": \"Creative, communicative\",
    \"availability_hours\": 35,
    \"timezone\": \"UTC+2\"
  },
  {
    \"id\": \"3\",
    \"name\": \"Alex Chen\",
    \"role\": \"Data Scientist\",
    \"skills\": [\"Python\", \"Machine Learning\", \"SQL\", \"TensorFlow\", \"PyTorch\"],
    \"skill_levels\": {
      \"Python\": \"Advanced\",
      \"Machine Learning\": \"Advanced\",
      \"SQL\": \"Advanced\", 
      \"TensorFlow\": \"Intermediate\",
      \"PyTorch\": \"Intermediate\"
    },
    \"experience_years\": 4,
    \"tools\": [\"Jupyter\", \"Scikit-learn\", \"Pandas\", \"NumPy\"],
    \"collaboration_style\": \"Analytical, structured\",
    \"availability_hours\": 40,
    \"timezone\": \"UTC+1\"
  }
]"""

create_file "data/sample/sample_projects.json" """[
  {
    \"id\": \"p1\",
    \"title\": \"E-commerce Platform Redesign\",
    \"description\": \"Redesign our e-commerce platform with modern UI/UX, improve performance, and add recommendation features.\",
    \"required_roles\": [\"Frontend Developer\", \"Backend Developer\", \"UX Designer\", \"Data Scientist\"],
    \"required_skills\": [\"React\", \"Python\", \"Figma\", \"Machine Learning\", \"SQL\"],
    \"team_size\": 4,
    \"duration_weeks\": 12,
    \"priority_skills\": [\"React\", \"Python\", \"Figma\"]
  },
  {
    \"id\": \"p2\", 
    \"title\": \"Mobile App Development\",
    \"description\": \"Develop a cross-platform mobile app for task management with cloud synchronization.\",
    \"required_roles\": [\"Mobile Developer\", \"Backend Developer\", \"UI Designer\"],
    \"required_skills\": [\"React Native\", \"Node.js\", \"MongoDB\", \"UI Design\"],
    \"team_size\": 3,
    \"duration_weeks\": 8,
    \"priority_skills\": [\"React Native\", \"Node.js\"]
  }
]"""

print_color "\n📚 Creating documentation..." "$BLUE"

# Create architecture documentation
create_file "docs/architecture.md" """# SkillSync Architecture

## System Overview

SkillSync is an AI-powered team formation system that automatically creates balanced, diverse, and complementary teams by analyzing skills, experience, and preferences.

## Architecture Diagram

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (M4)                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                    Integration Layer                        │
└─────┬──────────────┬──────────────┬─────────────┬──────────┘
      │              │              │             │
┌─────▼─────┐  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
│   JSON    │  │ Candidate │  │ Project  │  │  Team   │
│Extraction │  │Embeddings │  │Embeddings│  │  Builder │
│   (M5)    │  │   (M1)    │  │   (M2)   │  │   (M3)  │
└─────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
      │              │              │             │
      └──────────────┴──────────────┴─────────────┘
                               │
                     ┌─────────▼─────────┐
                     │    Explanations   │
                     │       (M5)        │
                     └───────────────────┘
\`\`\`

## Data Flow

1. **Input**: Raw profiles text or JSON (M5 processes if raw)
2. **Processing**: 
   - M5 extracts structured JSON from raw text
   - M1 generates embeddings for all candidates
   - M2 generates embedding for project description
   - M3 runs greedy algorithm using both embeddings
   - M5 generates explanations for formed teams
3. **Output**: Recommended teams with explanations in UI

## Module Responsibilities

### M1: Candidate Embeddings (Shahzad)
- Generate embeddings for each candidate profile
- Compute candidate-candidate similarity matrix
- Store embeddings in FAISS vector database

### M2: Project Embeddings (Roxana)  
- Generate embedding for project description
- Compute candidate-project similarity scores
- Rank candidates by project fit

### M3: Team Construction (Ana)
- Implement greedy algorithm for team formation
- Balance: fit, diversity, skill coverage
- Enforce constraints (roles, team size)

### M4: UI & Architecture (Șaban)
- Streamlit web interface
- System architecture and integration
- Git management and coordination

### M5: JSON Extraction & Explanations (Noor)
- Extract structured data from raw profiles (ChatGPT)
- Generate human-readable team explanations (ChatGPT)
- Validate JSON schema compliance

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.x
- **AI/ML**: 
  - OpenAI GPT (extraction & explanations)
  - Sentence Transformers (embeddings)
  - FAISS (vector similarity search)
- **Data**: JSON, CSV
- **Visualization**: Plotly, Matplotlib

## Interface Contracts

All modules communicate through defined interfaces in \`shared/interfaces.py\`:

1. \`extract_profiles_from_raw()\` - M5 → All
2. \`generate_candidate_embeddings()\` - M1 → M3, M4
3. \`generate_project_embedding()\` - M2 → M3, M4  
4. \`build_teams()\` - M3 → M4, M5
5. \`generate_explanations()\` - M5 → M4

## Testing Strategy

1. **Unit Tests**: Each member tests their own module
2. **Integration Tests**: Complete pipeline test
3. **End-to-End Tests**: Full system with sample data

## Deployment

Local development:
\`\`\`bash
pip install -r requirements.txt
streamlit run src/modules/member4_ui/app.py
\`\`\`

## Future Extensions

1. Multi-project team assignment
2. Learning from feedback
3. Integration with HR systems
4. Advanced visualization dashboards
"""

# Create development guide
create_file "docs/development_guide.md" """# Development Guide

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- OpenAI API key (for M5)

### Installation
\`\`\`bash
# Clone repository
git clone [your-repo-url]
cd skill-sync

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env
# Edit .env and add your OpenAI API key
\`\`\`

### Running the Application
\`\`\`bash
# Start Streamlit UI
streamlit run src/modules/member4_ui/app.py

# Run integration test
python tests/integration_test.py
\`\`\`

## Module Development

### Each Member's Workspace
Each member works in their own directory:
- \`src/modules/member1_candidate_embeddings/\` - Shahzad
- \`src/modules/member2_project_embeddings/\` - Roxana  
- \`src/modules/member3_team_builder/\` - Ana
- \`src/modules/member4_ui/\` - Șaban
- \`src/modules/member5_json_explainer/\` - Noor

### Implementing Your Module

1. **Understand the interface** in \`shared/interfaces.py\`
2. **Create your main module file** (e.g., \`embeddings.py\` for M1)
3. **Implement the required functions** from the interface
4. **Create tests** in your module directory
5. **Update requirements.txt** if you need new packages

### Example: M1 Creating Candidate Embeddings

\`\`\`python
# src/modules/member1_candidate_embeddings/embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
from shared.interfaces import CandidateProfile

class CandidateEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, profiles: List[CandidateProfile]):
        # Your implementation here
        pass
\`\`\`

## Git Workflow

### Branch Strategy
- \`main\`: Production-ready code
- \`develop\`: Integration branch
- \`feature/memberX-description\`: Feature branches

### Daily Workflow
1. Pull latest changes
\`\`\`bash
git checkout develop
git pull origin develop
\`\`\`

2. Create feature branch
\`\`\`bash
git checkout -b feature/m1-candidate-embeddings
\`\`\`

3. Work on your feature
4. Commit regularly
\`\`\`bash
git add .
git commit -m \"M1: Implemented candidate embedding generation\"
\`\`\`

5. Push to remote
\`\`\`bash
git push origin feature/m1-candidate-embeddings
\`\`\`

6. Create Pull Request to \`develop\`

### Commit Message Convention
Prefix with member identifier:
- \`M1:\` - Shahzad
- \`M2:\` - Roxana
- \`M3:\` - Ana  
- \`M4:\` - Șaban
- \`M5:\` - Noor

Examples:
\`\`\`
M1: Added cosine similarity computation
M4: Fixed UI layout issue
M5: Improved JSON extraction prompts
\`\`\`

## Testing

### Running Tests
\`\`\`bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_member1_embeddings.py

# Run with coverage
pytest --cov=src tests/
\`\`\`

### Creating Tests
Create test files in \`tests/\` directory:
- \`test_member1_embeddings.py\`
- \`test_member2_project_embeddings.py\`
- etc.

## Integration Points

### Weekly Integration Sessions
1. **Monday**: Plan week, assign tasks
2. **Wednesday**: Mid-week check, solve blockers
3. **Friday**: Integration test, prepare for demo

### Integration Checklist
Before integrating:
- [ ] Your module passes all tests
- [ ] You've implemented the interface functions
- [ ] You've updated requirements.txt if needed
- [ ] You've documented your module

## Common Issues & Solutions

### Module Import Errors
If you get import errors:
\`\`\`python
import sys
sys.path.append('path/to/skill-sync/src')
\`\`\`

### API Key Issues
Store API keys in \`.env\` file and load with:
\`\`\`python
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
\`\`\`

### Performance Issues
- Use FAISS for vector search (not brute force)
- Cache embeddings to disk
- Batch process when possible

## Getting Help

1. Check \`docs/\` directory first
2. Look at sample code in \`data/sample/\`
3. Ask in team chat
4. Create GitHub issue for bugs

## Code Quality Guidelines

1. **Documentation**: Docstrings for all functions
2. **Type Hints**: Use Python type annotations
3. **Error Handling**: Use try-except for external calls
4. **Logging**: Use logging module, not print statements
5. **Code Style**: Follow PEP 8, use Black formatter

Example:
\`\`\`python
def calculate_similarity(vec1: np.array, vec2: np.array) -> float:
    \"\"\"
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
        
    Raises:
        ValueError: If vectors have different dimensions
    \"\"\"
    if vec1.shape != vec2.shape:
        raise ValueError(\"Vectors must have same dimension\")
    
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
\`\`\`
"""

print_color "\n🔧 Creating module templates..." "$BLUE"

# Create M1 template
create_file "src/modules/candidate_embeddings/embeddings.py" """
\"\"\"
Candidate Embeddings Module (M1 - Shahzad)
\"\"\"

import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer
import faiss

from shared.interfaces import CandidateProfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CandidateEmbedder:
    \"\"\"Generates embeddings for candidate profiles\"\"\"
    
    def __init__(self, model_name: str = \"all-MiniLM-L6-v2\"):
        \"\"\"
        Initialize the embedder.
        
        Args:
            model_name: Name of sentence transformer model
        \"\"\"
        logger.info(f\"Loading model: {model_name}\")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.candidate_ids = []
        
    def profile_to_text(self, profile: CandidateProfile) -> str:
        \"\"\"
        Convert candidate profile to canonical text for embedding.
        
        Args:
            profile: CandidateProfile object
            
        Returns:
            Text representation of profile
        \"\"\"
        # Create canonical text template
        skills_text = \", \".join([f\"{skill}\" for skill in profile.skills])
        tools_text = \", \".join(profile.tools) if profile.tools else \"\"
        
        text_parts = [
            f\"Role: {profile.role}\",
            f\"Skills: {skills_text}\",
            f\"Experience: {profile.experience_years} years\",
            f\"Collaboration style: {profile.collaboration_style}\",
            f\"Availability: {profile.availability_hours} hours/week\"
        ]
        
        if tools_text:
            text_parts.append(f\"Tools: {tools_text}\")
            
        return \". \".join(text_parts)
    
    def generate_embeddings(self, profiles: List[CandidateProfile]) -> Dict[str, Any]:
        \"\"\"
        Generate embeddings for all candidates and compute similarity matrix.
        
        Args:
            profiles: List of CandidateProfile objects
            
        Returns:
            Dictionary with embeddings, similarity matrix, and candidate IDs
        \"\"\"
        logger.info(f\"Generating embeddings for {len(profiles)} candidates\")
        
        # Convert profiles to text
        texts = [self.profile_to_text(profile) for profile in profiles]
        self.candidate_ids = [profile.id for profile in profiles]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index for efficient similarity search
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Compute similarity matrix (cosine similarity)
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Store embeddings in dictionary
        embedding_dict = {
            profile.id: embedding for profile, embedding in zip(profiles, embeddings)
        }
        
        result = {
            'embeddings': embedding_dict,
            'similarity_matrix': similarity_matrix,
            'candidate_ids': self.candidate_ids,
            'faiss_index': self.index,
            'normalized_embeddings': normalized_embeddings
        }
        
        logger.info(f\"Embeddings generated. Dimension: {embeddings.shape}\")
        return result
    
    def get_similar_candidates(self, candidate_id: str, top_k: int = 5) -> List[Dict]:
        \"\"\"
        Find candidates most similar to the given candidate.
        
        Args:
            candidate_id: ID of candidate to find similar ones for
            top_k: Number of similar candidates to return
            
        Returns:
            List of similar candidates with similarity scores
        \"\"\"
        if self.index is None or candidate_id not in self.embeddings:
            raise ValueError(\"Embeddings not generated or candidate not found\")
        
        # Get candidate embedding
        candidate_idx = self.candidate_ids.index(candidate_id)
        candidate_embedding = self.normalized_embeddings[candidate_idx].reshape(1, -1)
        
        # Search for similar candidates
        distances, indices = self.index.search(candidate_embedding.astype('float32'), top_k + 1)
        
        # Convert distances to similarities (cosine similarity from L2 distance)
        # For normalized vectors: similarity = 1 - distance^2 / 2
        similarities = 1 - (distances[0] ** 2) / 2
        
        # Prepare results (skip the first one as it's the candidate itself)
        results = []
        for i in range(1, min(top_k + 1, len(indices[0]))):
            idx = indices[0][i]
            if idx < len(self.candidate_ids):
                results.append({
                    'candidate_id': self.candidate_ids[idx],
                    'similarity': float(similarities[i])
                })
        
        return results

# Interface function
def generate_candidate_embeddings(profiles: List[CandidateProfile]) -> Dict[str, Any]:
    \"\"\"
    Interface function for M1 module.
    
    Args:
        profiles: List of CandidateProfile objects
        
    Returns:
        Dictionary with embeddings and similarity data
    \"\"\"
    embedder = CandidateEmbedder()
    return embedder.generate_embeddings(profiles)

if __name__ == \"__main__\":
    \"\"\"Test the embedder with sample data\"\"\"
    # Load sample data
    sample_file = Path(__file__).parent.parent.parent.parent / \"data\" / \"sample\" / \"sample_profiles.json\"
    with open(sample_file) as f:
        profiles_data = json.load(f)
    
    # Convert to CandidateProfile objects (simplified)
    profiles = []
    for i, data in enumerate(profiles_data[:3]):
        profile = CandidateProfile(
            id=data.get(\"id\", str(i)),
            name=data.get(\"name\", \"\"),
            skills=data.get(\"skills\", []),
            skill_levels=data.get(\"skill_levels\", {}),
            role=data.get(\"role\", \"\"),
            experience_years=data.get(\"experience_years\", 0),
            collaboration_style=data.get(\"collaboration_style\", \"\"),
            availability_hours=data.get(\"availability_hours\", 40),
            tools=data.get(\"tools\", []),
            domains=data.get(\"domains\", [])
        )
        profiles.append(profile)
    
    # Generate embeddings
    result = generate_candidate_embeddings(profiles)
    print(f\"Generated embeddings for {len(profiles)} candidates\")
    print(f\"Similarity matrix shape: {result['similarity_matrix'].shape}\")
    print(f\"Sample similarity between first two: {result['similarity_matrix'][0, 1]:.3f}\")
"""

# Create M2 template
create_file "src/modules/project_embeddings/embeddings.py" """
\"\"\"
Project Embeddings Module (M2 - Roxana)
\"\"\"

import numpy as np
from typing import List, Dict, Any
import json
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

from shared.interfaces import ProjectDescription, CandidateProfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectEmbedder:
    \"\"\"Generates embeddings for project descriptions\"\"\"
    
    def __init__(self, model_name: str = \"all-MiniLM-L6-v2\"):
        \"\"\"
        Initialize the project embedder.
        
        Args:
            model_name: Name of sentence transformer model
        \"\"\"
        logger.info(f\"Loading model: {model_name}\")
        self.model = SentenceTransformer(model_name)
        
    def project_to_text(self, project: ProjectDescription) -> str:
        \"\"\"
        Convert project description to canonical text for embedding.
        
        Args:
            project: ProjectDescription object
            
        Returns:
            Text representation of project
        \"\"\"
        # Create canonical text template
        roles_text = \", \".join(project.required_roles)
        skills_text = \", \".join(project.required_skills)
        
        text_parts = [
            f\"Project: {project.title}\",
            f\"Description: {project.description}\",
            f\"Required roles: {roles_text}\",
            f\"Required skills: {skills_text}\",
            f\"Team size: {project.team_size}\",
            f\"Duration: {project.duration_weeks} weeks\"
        ]
        
        if project.priority_skills:
            priority_text = \", \".join(project.priority_skills)
            text_parts.append(f\"Priority skills: {priority_text}\")
            
        return \". \".join(text_parts)
    
    def generate_project_embedding(self, project: ProjectDescription) -> np.array:
        \"\"\"
        Generate embedding for a project description.
        
        Args:
            project: ProjectDescription object
            
        Returns:
            Project embedding vector
        \"\"\"
        logger.info(f\"Generating embedding for project: {project.title}\")
        
        # Convert project to text
        text = self.project_to_text(project)
        
        # Generate embedding
        embedding = self.model.encode(text, show_progress_bar=False)
        
        logger.info(f\"Project embedding generated. Dimension: {embedding.shape}\")
        return embedding
    
    def compute_candidate_project_similarities(
        self, 
        project_embedding: np.array,
        candidate_embeddings: Dict[str, np.array]
    ) -> Dict[str, float]:
        \"\"\"
        Compute cosine similarity between project and all candidates.
        
        Args:
            project_embedding: Project embedding vector
            candidate_embeddings: Dictionary of candidate_id -> embedding
            
        Returns:
            Dictionary of candidate_id -> similarity score
        \"\"\"
        logger.info(f\"Computing candidate-project similarities for {len(candidate_embeddings)} candidates\")
        
        # Normalize project embedding
        project_norm = np.linalg.norm(project_embedding)
        if project_norm == 0:
            project_normalized = project_embedding
        else:
            project_normalized = project_embedding / project_norm
        
        similarities = {}
        
        for candidate_id, candidate_embedding in candidate_embeddings.items():
            # Normalize candidate embedding
            candidate_norm = np.linalg.norm(candidate_embedding)
            if candidate_norm == 0:
                candidate_normalized = candidate_embedding
            else:
                candidate_normalized = candidate_embedding / candidate_norm
            
            # Compute cosine similarity
            similarity = float(np.dot(project_normalized, candidate_normalized))
            similarities[candidate_id] = similarity
        
        logger.info(f\"Similarities computed. Range: [{min(similarities.values()):.3f}, {max(similarities.values()):.3f}]\")
        return similarities
    
    def get_top_candidates(
        self, 
        similarities: Dict[str, float], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        \"\"\"
        Get top-k candidates by project similarity.
        
        Args:
            similarities: Dictionary of candidate_id -> similarity
            top_k: Number of top candidates to return
            
        Returns:
            List of top candidates with scores, sorted descending
        \"\"\"
        # Sort candidates by similarity
        sorted_candidates = sorted(
            similarities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Format results
        results = [
            {\"candidate_id\": cand_id, \"similarity\": score}
            for cand_id, score in sorted_candidates
        ]
        
        return results

# Interface function
def generate_project_embedding(project: ProjectDescription) -> Dict[str, Any]:
    \"\"\"
    Interface function for M2 module.
    
    Args:
        project: ProjectDescription object
        
    Returns:
        Dictionary with project embedding (placeholder - will be extended)
    \"\"\"
    embedder = ProjectEmbedder()
    project_embedding = embedder.generate_project_embedding(project)
    
    return {
        'project_embedding': project_embedding,
        'project_id': project.id,
        'project_title': project.title
    }

if __name__ == \"__main__\":
    \"\"\"Test the project embedder with sample data\"\"\"
    # Load sample data
    sample_file = Path(__file__).parent.parent.parent.parent / \"data\" / \"sample\" / \"sample_projects.json\"
    with open(sample_file) as f:
        projects_data = json.load(f)
    
    # Convert to ProjectDescription object
    project_data = projects_data[0]
    project = ProjectDescription(
        id=project_data[\"id\"],
        title=project_data[\"title\"],
        description=project_data[\"description\"],
        required_roles=project_data[\"required_roles\"],
        required_skills=project_data[\"required_skills\"],
        team_size=project_data[\"team_size\"],
        duration_weeks=project_data[\"duration_weeks\"],
        priority_skills=project_data[\"priority_skills\"]
    )
    
    # Generate project embedding
    result = generate_project_embedding(project)
    print(f\"Generated embedding for project: {project.title}\")
    print(f\"Embedding shape: {result['project_embedding'].shape}\")
"""

# Create M3 template
create_file "src/modules/team_builder/builder.py" """
\"\"\"
Team Construction Module (M3 - Ana)
\"\"\"

import numpy as np
from typing import List, Dict, Any
import logging
from shared.interfaces import CandidateProfile, ProjectDescription

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TeamBuilder:
    \"\"\"Greedy algorithm for team formation\"\"\"
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        \"\"\"
        Initialize team builder with weights.
        
        Args:
            alpha: Weight for candidate-project fit
            beta: Weight for diversity (1 - similarity)
            gamma: Weight for skill coverage
        \"\"\"
        self.alpha = alpha  # Fit weight
        self.beta = beta    # Diversity weight  
        self.gamma = gamma  # Coverage weight
        
    def calculate_skill_coverage(self, team: List[CandidateProfile], 
                               required_skills: List[str]) -> float:
        \"\"\"
        Calculate what percentage of required skills are covered by the team.
        
        Args:
            team: List of candidate profiles
            required_skills: List of skills required for the project
            
        Returns:
            Coverage score between 0 and 1
        \"\"\"
        if not required_skills:
            return 1.0
            
        # Get all unique skills from team members
        team_skills = set()
        for member in team:
            team_skills.update(member.skills)
        
        # Calculate coverage
        covered_skills = set(required_skills) & team_skills
        coverage = len(covered_skills) / len(required_skills)
        
        return coverage
    
    def calculate_team_diversity(self, team: List[CandidateProfile],
                               candidate_sim_matrix: np.array,
                               candidate_ids: List[str]) -> float:
        \"\"\"
        Calculate diversity score for a team (1 - average similarity).
        
        Args:
            team: List of candidate profiles
            candidate_sim_matrix: Precomputed similarity matrix
            candidate_ids: List of candidate IDs in order of matrix
            
        Returns:
            Diversity score between 0 and 1 (higher = more diverse)
        \"\"\"
        if len(team) <= 1:
            return 1.0
            
        # Get indices of team members in similarity matrix
        team_indices = []
        for member in team:
            if member.id in candidate_ids:
                idx = candidate_ids.index(member.id)
                team_indices.append(idx)
        
        if len(team_indices) <= 1:
            return 1.0
            
        # Calculate average similarity between team members
        total_similarity = 0
        count = 0
        
        for i in range(len(team_indices)):
            for j in range(i + 1, len(team_indices)):
                idx_i = team_indices[i]
                idx_j = team_indices[j]
                total_similarity += candidate_sim_matrix[idx_i, idx_j]
                count += 1
        
        if count == 0:
            return 1.0
            
        avg_similarity = total_similarity / count
        
        # Diversity is inverse of similarity
        diversity = 1 - avg_similarity
        
        return max(0, min(1, diversity))
    
    def calculate_team_fit(self, team: List[CandidateProfile],
                         project_similarities: Dict[str, float]) -> float:
        \"\"\"
        Calculate average project fit for team members.
        
        Args:
            team: List of candidate profiles
            project_similarities: Dictionary of candidate_id -> similarity score
            
        Returns:
            Average fit score between 0 and 1
        \"\"\"
        if not team:
            return 0.0
            
        total_fit = 0
        count = 0
        
        for member in team:
            if member.id in project_similarities:
                total_fit += project_similarities[member.id]
                count += 1
        
        if count == 0:
            return 0.0
            
        return total_fit / count
    
    def calculate_team_score(self, team: List[CandidateProfile],
                           candidate_sim_matrix: np.array,
                           candidate_ids: List[str],
                           project_similarities: Dict[str, float],
                           required_skills: List[str]) -> float:
        \"\"\"
        Calculate overall team score using weighted combination.
        
        Args:
            team: List of candidate profiles
            candidate_sim_matrix: Precomputed similarity matrix
            candidate_ids: List of candidate IDs in order of matrix
            project_similarities: Dictionary of candidate_id -> similarity
            required_skills: List of required skills for project
            
        Returns:
            Overall team score
        \"\"\"
        # Calculate individual components
        coverage = self.calculate_skill_coverage(team, required_skills)
        diversity = self.calculate_team_diversity(team, candidate_sim_matrix, candidate_ids)
        fit = self.calculate_team_fit(team, project_similarities)
        
        # Weighted combination
        score = (self.alpha * fit + 
                self.beta * diversity + 
                self.gamma * coverage)
        
        return score
    
    def build_team_greedy(self,
                         profiles: List[CandidateProfile],
                         project: ProjectDescription,
                         candidate_sim_matrix: np.array,
                         candidate_ids: List[str],
                         project_similarities: Dict[str, float]) -> List[CandidateProfile]:
        \"\"\"
        Build a team using greedy algorithm.
        
        Args:
            profiles: List of all candidate profiles
            project: Project description
            candidate_sim_matrix: Precomputed similarity matrix
            candidate_ids: List of candidate IDs in order of matrix
            project_similarities: Dictionary of candidate_id -> similarity
            
        Returns:
            List of selected team members
        \"\"\"
        logger.info(f\"Building team of size {project.team_size}\")
        
        # Filter out candidates with very low project similarity
        filtered_profiles = []
        for profile in profiles:
            if profile.id in project_similarities and project_similarities[profile.id] > 0.3:
                filtered_profiles.append(profile)
        
        if not filtered_profiles:
            filtered_profiles = profiles[:project.team_size]
            logger.warning(\"No candidates passed similarity filter, using first few\")
        
        # Start with candidate with highest project fit
        sorted_by_fit = sorted(filtered_profiles, 
                             key=lambda p: project_similarities.get(p.id, 0), 
                             reverse=True)
        
        if not sorted_by_fit:
            return []
        
        team = [sorted_by_fit[0]]
        remaining = [p for p in filtered_profiles if p.id != sorted_by_fit[0].id]
        
        # Greedily add candidates until team size reached
        while len(team) < project.team_size and remaining:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Try adding this candidate
                potential_team = team + [candidate]
                
                # Calculate score for this potential team
                score = self.calculate_team_score(
                    potential_team,
                    candidate_sim_matrix,
                    candidate_ids,
                    project_similarities,
                    project.required_skills
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                team.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        logger.info(f\"Team built with {len(team)} members. Score: {best_score:.3f}\")
        return team

# Interface function
def build_teams(
    profiles: List[CandidateProfile],
    project: ProjectDescription,
    candidate_sim_matrix: np.array,
    candidate_ids: List[str],
    project_similarities: Dict[str, float]
) -> List[List[CandidateProfile]]:
    \"\"\"
    Interface function for M3 module.
    
    Args:
        profiles: List of candidate profiles
        project: Project description
        candidate_sim_matrix: Precomputed similarity matrix
        candidate_ids: List of candidate IDs in order of matrix
        project_similarities: Dictionary of candidate_id -> similarity score
        
    Returns:
        List of teams (each team is list of CandidateProfile objects)
    \"\"\"
    builder = TeamBuilder()
    
    # For now, return one team - can be extended to multiple teams
    team = builder.build_team_greedy(
        profiles, project, candidate_sim_matrix, candidate_ids, project_similarities
    )
    
    return [team] if team else []

if __name__ == \"__main__\":
    \"\"\"Test the team builder with sample data\"\"\"
    # Create mock data for testing
    profiles = [
        CandidateProfile(
            id=\"1\",
            name=\"Alice\",
            skills=[\"Python\", \"ML\"],
            skill_levels={\"Python\": \"Advanced\", \"ML\": \"Intermediate\"},
            role=\"Data Scientist\",
            experience_years=3,
            collaboration_style=\"Analytical\",
            availability_hours=40,
            tools=[],
            domains=[]
        ),
        CandidateProfile(
            id=\"2\",
            name=\"Bob\",
            skills=[\"React\", \"JavaScript\"],
            skill_levels={\"React\": \"Expert\", \"JavaScript\": \"Advanced\"},
            role=\"Frontend Developer\",
            experience_years=2,
            collaboration_style=\"Creative\",
            availability_hours=35,
            tools=[],
            domains=[]
        )
    ]
    
    project = ProjectDescription(
        id=\"p1\",
        title=\"Test Project\",
        description=\"A test project\",
        required_roles=[\"Data Scientist\", \"Frontend Developer\"],
        required_skills=[\"Python\", \"React\"],
        team_size=2,
        duration_weeks=4,
        priority_skills=[]
    )
    
    # Mock similarity data
    candidate_sim_matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    candidate_ids = [\"1\", \"2\"]
    project_similarities = {\"1\": 0.8, \"2\": 0.9}
    
    # Build team
    teams = build_teams(profiles, project, candidate_sim_matrix, candidate_ids, project_similarities)
    
    print(f\"Built {len(teams)} team(s)\")
    for i, team in enumerate(teams):
        print(f\"Team {i+1} members: {[member.name for member in team]}\")
"""

# Create M5 template for JSON extraction
create_file "src/modules/json_explainer/extractor.py" """
# Continuing from where the file left off...
\"\"\"
JSON Extraction Module (M5 - Noor)
\"\"\"

import json
from typing import List, Dict, Any
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from pathlib import Path

from shared.interfaces import CandidateProfile

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONExtractor:
    \"\"\"Extract structured JSON from raw text using ChatGPT API\"\"\"
    
    def __init__(self, model: str = \"gpt-3.5-turbo\"):
        \"\"\"
        Initialize the JSON extractor.
        
        Args:
            model: OpenAI model to use
        \"\"\"
        api_key = os.getenv(\"OPENAI_API_KEY\")
        if not api_key:
            raise ValueError(\"OPENAI_API_KEY not found in environment variables\")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
    def create_extraction_prompt(self, raw_text: str) -> str:
        \"\"\"
        Create prompt for extracting structured data from raw profiles.
        
        Args:
            raw_text: Raw text containing one or more profiles
            
        Returns:
            Prompt for ChatGPT
        \"\"\"
        prompt = f\"\"\"Extract structured information from the following candidate profiles. 
        For each candidate, extract:
        1. Name
        2. Role/title
        3. List of skills with levels (beginner, intermediate, advanced, expert)
        4. Years of experience (extract numeric value)
        5. Tools/technologies used
        6. Collaboration style/preferences
        7. Availability (hours per week, extract numeric value)
        8. Timezone (if mentioned)
        
        Return the data as a JSON array of objects with this structure:
        [
          {{
            \"id\": \"unique_id_1\",
            \"name\": \"Full Name\",
            \"role\": \"Job Role\",
            \"skills\": [\"Skill1\", \"Skill2\", \"Skill3\"],
            \"skill_levels\": {{\"Skill1\": \"advanced\", \"Skill2\": \"intermediate\"}},
            \"experience_years\": 5,
            \"tools\": [\"Tool1\", \"Tool2\"],
            \"collaboration_style\": \"description\",
            \"availability_hours\": 40,
            \"timezone\": \"UTC+1\",
            \"domains\": [\"Domain1\", \"Domain2\"] if mentioned
          }}
        ]
        
        IMPORTANT RULES:
        - Generate unique sequential IDs starting from \"1\"
        - If skill levels are not mentioned, use \"intermediate\" as default
        - If availability is not mentioned, use 40 as default
        - If experience is not mentioned, use 0
        - Extract only the information explicitly mentioned
        - Do not make up or infer information
        
        Profiles to process:
        {raw_text}
        \"\"\"
        return prompt
    
    def extract_profiles(self, raw_text: str) -> List[Dict[str, Any]]:
        \"\"\"
        Extract structured profiles from raw text using OpenAI API.
        
        Args:
            raw_text: Raw text containing candidate profiles
            
        Returns:
            List of structured profile dictionaries
        \"\"\"
        logger.info(\"Extracting structured profiles from raw text\")
        
        try:
            # Create prompt
            prompt = self.create_extraction_prompt(raw_text)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {\"role\": \"system\", \"content\": \"You are a helpful assistant that extracts structured data from text. Always return valid JSON.\"},
                    {\"role\": \"user\", \"content\": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                response_format={\"type\": \"json_object\"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract profiles array
            profiles = result.get(\"profiles\", result)  # Handle different response formats
            
            if not isinstance(profiles, list):
                profiles = [profiles]
            
            logger.info(f\"Successfully extracted {len(profiles)} profiles\")
            return profiles
            
        except Exception as e:
            logger.error(f\"Error extracting profiles: {e}\")
            # Fallback to simple parsing
            return self._fallback_extraction(raw_text)
    
    def _fallback_extraction(self, raw_text: str) -> List[Dict[str, Any]]:
        \"\"\"
        Fallback extraction method if API call fails.
        Uses simple regex patterns to extract basic information.
        
        Args:
            raw_text: Raw text containing candidate profiles
            
        Returns:
            List of basic profile dictionaries
        \"\"\"
        logger.warning(\"Using fallback extraction method\")
        
        # Split text into candidate sections
        sections = re.split(r'\n\s*\n', raw_text.strip())
        profiles = []
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            profile = {
                \"id\": str(i + 1),
                \"name\": \"\",
                \"role\": \"\",
                \"skills\": [],
                \"skill_levels\": {},
                \"experience_years\": 0,
                \"tools\": [],
                \"collaboration_style\": \"\",
                \"availability_hours\": 40,
                \"timezone\": \"\",
                \"domains\": []
            }
            
            # Extract name (first line)
            lines = section.strip().split('\n')
            if lines:
                profile[\"name\"] = lines[0].strip()
            
            # Extract skills (simplified)
            for line in lines:
                line_lower = line.lower()
                if 'skill' in line_lower:
                    # Simple skill extraction - in real implementation, use more sophisticated parsing
                    pass
            
            profiles.append(profile)
        
        return profiles
    
    def validate_profile(self, profile: Dict[str, Any]) -> bool:
        \"\"\"
        Validate extracted profile against schema.
        
        Args:
            profile: Profile dictionary to validate
            
        Returns:
            True if valid, False otherwise
        \"\"\"
        required_fields = [\"id\", \"name\", \"role\", \"skills\"]
        
        for field in required_fields:
            if field not in profile:
                logger.warning(f\"Missing required field: {field}\")
                return False
        
        # Validate types
        if not isinstance(profile.get(\"skills\", []), list):
            logger.warning(\"Skills should be a list\")
            return False
        
        if not isinstance(profile.get(\"experience_years\", 0), (int, float)):
            logger.warning(\"experience_years should be a number\")
            return False
        
        return True

# Interface function
def extract_profiles_from_raw(raw_text: str) -> List[CandidateProfile]:
    \"\"\"
    Interface function for M5 module.
    
    Args:
        raw_text: Raw text containing candidate profiles
        
    Returns:
        List of CandidateProfile objects
    \"\"\"
    extractor = JSONExtractor()
    
    # Extract structured data
    profile_dicts = extractor.extract_profiles(raw_text)
    
    # Convert to CandidateProfile objects
    profiles = []
    for profile_dict in profile_dicts:
        # Validate profile
        if not extractor.validate_profile(profile_dict):
            logger.warning(f\"Skipping invalid profile: {profile_dict.get('name', 'Unknown')}\")
            continue
        
        try:
            profile = CandidateProfile(
                id=profile_dict.get(\"id\", \"\"),
                name=profile_dict.get(\"name\", \"\"),
                skills=profile_dict.get(\"skills\", []),
                skill_levels=profile_dict.get(\"skill_levels\", {}),
                role=profile_dict.get(\"role\", \"\"),
                experience_years=profile_dict.get(\"experience_years\", 0),
                collaboration_style=profile_dict.get(\"collaboration_style\", \"\"),
                availability_hours=profile_dict.get(\"availability_hours\", 40),
                tools=profile_dict.get(\"tools\", []),
                domains=profile_dict.get(\"domains\", [])
            )
            profiles.append(profile)
        except Exception as e:
            logger.error(f\"Error converting profile: {e}\")
    
    return profiles

if __name__ == \"__main__\":
    \"\"\"Test the JSON extractor with sample data\"\"\"
    # Load sample data
    sample_file = Path(__file__).parent.parent.parent.parent / \"data\" / \"sample\" / \"sample_profiles_raw.txt\"
    with open(sample_file, 'r') as f:
        raw_text = f.read()
    
    # Extract profiles
    profiles = extract_profiles_from_raw(raw_text)
    
    print(f\"Extracted {len(profiles)} profiles:\")
    for i, profile in enumerate(profiles[:3]):  # Show first 3
        print(f\"\nProfile {i+1}:\")
        print(f\"  Name: {profile.name}\")
        print(f\"  Role: {profile.role}\")
        print(f\"  Skills: {', '.join(profile.skills[:5])}\")
        print(f\"  Experience: {profile.experience_years} years\")
"""

# Create M5 explanations module
create_file "src/modules/json_explainer/explanations.py" """
\"\"\"
Team Explanations Module (M5 - Noor)
\"\"\"

import json
from typing import List, Dict, Any
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

from shared.interfaces import CandidateProfile, ProjectDescription

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExplanationGenerator:
    \"\"\"Generate human-readable explanations for teams using ChatGPT\"\"\"
    
    def __init__(self, model: str = \"gpt-3.5-turbo\"):
        \"\"\"
        Initialize the explanation generator.
        
        Args:
            model: OpenAI model to use
        \"\"\"
        api_key = os.getenv(\"OPENAI_API_KEY\")
        if not api_key:
            raise ValueError(\"OPENAI_API_KEY not found in environment variables\")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def create_team_summary(self, team: List[CandidateProfile]) -> str:
        \"\"\"
        Create a textual summary of the team for the prompt.
        
        Args:
            team: List of team members
            
        Returns:
            Textual summary of the team
        \"\"\"
        summary_parts = []
        
        for member in team:
            member_summary = f\"{member.name} ({member.role}):\"
            member_summary += f\"\\n  - Skills: {', '.join(member.skills[:5])}\"
            if member.skill_levels:
                top_skills = list(member.skill_levels.items())[:3]
                member_summary += f\"\\n  - Top skill levels: {', '.join([f'{k}: {v}' for k, v in top_skills])}\"
            member_summary += f\"\\n  - Experience: {member.experience_years} years\"
            member_summary += f\"\\n  - Collaboration style: {member.collaboration_style}\"
            member_summary += f\"\\n  - Availability: {member.availability_hours} hours/week\"
            summary_parts.append(member_summary)
        
        return \"\\n\\n\".join(summary_parts)
    
    def create_explanation_prompt(self, team: List[CandidateProfile], 
                                project: ProjectDescription) -> str:
        \"\"\"
        Create prompt for generating team explanations.
        
        Args:
            team: List of team members
            project: Project description
            
        Returns:
            Prompt for ChatGPT
        \"\"\"
        team_summary = self.create_team_summary(team)
        
        prompt = f\"\"\"Analyze the following team composition for the given project and provide a comprehensive explanation.

PROJECT:
Title: {project.title}
Description: {project.description}
Required Roles: {', '.join(project.required_roles)}
Required Skills: {', '.join(project.required_skills)}
Team Size: {project.team_size}
Duration: {project.duration_weeks} weeks

TEAM COMPOSITION:
{team_summary}

Please provide a detailed explanation covering:

1. **Overall Assessment**: How well does this team match the project requirements?

2. **Strengths**: 
   - Skill coverage for required skills
   - Role fulfillment
   - Complementary expertise
   - Collaboration potential

3. **Potential Gaps or Risks**:
   - Missing skills or roles
   - Possible collaboration challenges
   - Availability constraints
   - Experience level considerations

4. **Recommendations**:
   - Suggestions for team success
   - Areas to monitor
   - Potential need for additional support

Guidelines:
- Be specific and data-driven
- Reference actual skills and experience mentioned
- Avoid generic statements
- Focus on factual analysis based on provided data
- Keep explanation concise but comprehensive
- Do not make up information not provided

Format the response as a clear, well-structured paragraph suitable for a project manager.
\"\"\"
        
        return prompt
    
    def generate_explanation(self, team: List[CandidateProfile], 
                           project: ProjectDescription) -> str:
        \"\"\"
        Generate explanation for a single team.
        
        Args:
            team: List of team members
            project: Project description
            
        Returns:
            Human-readable explanation
        \"\"\"
        logger.info(f\"Generating explanation for team with {len(team)} members\")
        
        try:
            prompt = self.create_explanation_prompt(team, project)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {\"role\": \"system\", \"content\": \"You are an expert team composition analyst. Provide insightful, data-driven analysis of team formations.\"},
                    {\"role\": \"user\", \"content\": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info(\"Explanation generated successfully\")
            return explanation
            
        except Exception as e:
            logger.error(f\"Error generating explanation: {e}\")
            return self._generate_fallback_explanation(team, project)
    
    def _generate_fallback_explanation(self, team: List[CandidateProfile],
                                     project: ProjectDescription) -> str:
        \"\"\"
        Generate a simple fallback explanation without API.
        
        Args:
            team: List of team members
            project: Project description
            
        Returns:
            Basic explanation
        \"\"\"
        # Calculate basic metrics
        all_skills = []
        for member in team:
            all_skills.extend(member.skills)
        
        unique_skills = set(all_skills)
        required_skills = set(project.required_skills)
        covered_skills = required_skills.intersection(unique_skills)
        
        coverage_percentage = len(covered_skills) / len(required_skills) * 100 if required_skills else 100
        
        # Create simple explanation
        explanation = f\"This team consists of {len(team)} members covering {len(covered_skills)} out of {len(required_skills)} required skills ({coverage_percentage:.1f}%). \"
        explanation += f\"Team includes: {', '.join([member.role for member in team])}. \"
        
        if coverage_percentage > 80:
            explanation += \"The team has strong skill coverage for this project.\"
        elif coverage_percentage > 60:
            explanation += \"The team covers most required skills, but some gaps may need attention.\"
        else:
            explanation += \"The team has significant skill gaps that should be addressed.\"
        
        return explanation

# Interface function
def generate_explanations(
    teams: List[List[CandidateProfile]],
    project: ProjectDescription
) -> List[Dict[str, Any]]:
    \"\"\"
    Interface function for M5 module.
    
    Args:
        teams: List of teams (each team is list of CandidateProfile)
        project: Project description
        
    Returns:
        List of explanation objects for each team
    \"\"\"
    generator = ExplanationGenerator()
    explanations = []
    
    for i, team in enumerate(teams):
        explanation_text = generator.generate_explanation(team, project)
        
        explanation_obj = {
            \"team_id\": i + 1,
            \"team_members\": [member.name for member in team],
            \"explanation\": explanation_text,
            \"summary\": f\"Team {i+1}: {', '.join([member.role for member in team])}\"
        }
        
        explanations.append(explanation_obj)
    
    return explanations

if __name__ == \"__main__\":
    \"\"\"Test the explanation generator with sample data\"\"\"
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from shared.interfaces import CandidateProfile, ProjectDescription
    
    # Create sample team
    team = [
        CandidateProfile(
            id=\"1\",
            name=\"John Doe\",
            skills=[\"Python\", \"Django\", \"PostgreSQL\"],
            skill_levels={\"Python\": \"Expert\", \"Django\": \"Advanced\"},
            role=\"Backend Developer\",
            experience_years=5,
            collaboration_style=\"Agile, mentoring\",
            availability_hours=40,
            tools=[\"Git\", \"Docker\"],
            domains=[]
        ),
        CandidateProfile(
            id=\"2\",
            name=\"Jane Smith\",
            skills=[\"React\", \"JavaScript\", \"Figma\"],
            skill_levels={\"React\": \"Expert\", \"Figma\": \"Advanced\"},
            role=\"Frontend Developer\",
            experience_years=3,
            collaboration_style=\"Creative, communicative\",
            availability_hours=35,
            tools=[\"VS Code\", \"Git\"],
            domains=[]
        )
    ]
    
    # Create sample project
    project = ProjectDescription(
        id=\"p1\",
        title=\"Web Application Development\",
        description=\"Build a modern web application with frontend and backend components\",
        required_roles=[\"Backend Developer\", \"Frontend Developer\"],
        required_skills=[\"Python\", \"Django\", \"React\", \"JavaScript\"],
        team_size=2,
        duration_weeks=8,
        priority_skills=[\"Python\", \"React\"]
    )
    
    # Generate explanation
    explanations = generate_explanations([team], project)
    
    print(\"Generated Explanation:\")
    print(\"=\" * 50)
    print(explanations[0][\"explanation\"])
    print(\"=\" * 50)
"""

# create skill-sync/main.py
create_file "main.py" """
\"\"\"
Main orchestrator for SkillSync system
\"\"\"

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from shared.interfaces import extract_profiles_from_raw, generate_candidate_embeddings, generate_project_embedding, build_teams, generate_explanations
import json

def main():
    \"\"\"
    Main pipeline execution
    \"\"\"
    print('🚀 Starting SkillSync Pipeline')
    
    # 1. Load data
    # 2. Extract profiles
    # 3. Generate embeddings
    # 4. Build teams
    # 5. Generate explanations
    # 6. Output results
    
    print('✅ Pipeline completed')

if __name__ == '__main__':
    main()
"""

