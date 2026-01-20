
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
    page_title="SkillSync - AI Team Formation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
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
""", unsafe_allow_html=True)

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
st.markdown("<h1 class='main-header'>🎯 SkillSync</h1>", unsafe_allow_html=True)
st.markdown("### AI-Powered Team Formation System")

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/teamwork.png", width=80)
    st.markdown("### Navigation")
    
    page = st.radio(
        "Go to",
        ["📤 Upload Data", "⚙️ Project Setup", "👥 Build Teams", "📊 Analytics", "📋 Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Team Status")
    
    # Display module status
    status_cols = st.columns(2)
    with status_cols[0]:
        st.metric("Profiles", len(st.session_state.profiles) if st.session_state.profiles else "0")
    with status_cols[1]:
        st.metric("Teams", len(st.session_state.teams))
    
    st.markdown("---")
    st.markdown("**Team Members:**")
    st.markdown("""
    - Shahzad (M1): Candidate Embeddings
    - Roxana (M2): Project Embeddings  
    - Ana (M3): Team Algorithm
    - Noor (M5): JSON & Explanations
    """)

# Page 1: Upload Data
if page == "📤 Upload Data":
    st.header("📤 Upload Candidate Profiles")
    
    tab1, tab2 = st.tabs(["Upload JSON", "Raw Text Input"])
    
    with tab1:
        st.markdown("Upload structured JSON profiles (from M5's extraction)")
        uploaded_file = st.file_uploader("Choose JSON file", type=['json'])
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                # Convert to CandidateProfile objects
                # This is a placeholder - will be replaced by M5's actual extraction
                st.session_state.profiles = data
                st.success(f"✅ Successfully loaded {len(data)} profiles")
                
                # Preview
                with st.expander("Preview Profiles"):
                    df = pd.DataFrame(data[:5])
                    st.dataframe(df)
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab2:
        st.markdown("Paste raw profile text for M5 to process:")
        raw_text = st.text_area("Raw profiles (one per line):", height=200,
                               placeholder="John Doe\nSkills: Python, React, SQL\nRole: Full Stack Developer\n...")
        
        if st.button("Extract to JSON"):
            if raw_text:
                st.info("🔄 This will call M5's JSON extraction module")
                # Placeholder for M5's function
                # extracted = extract_profiles_from_raw(raw_text)
                st.session_state.profiles = [{"name": "Sample", "skills": ["Python"]}]  # Placeholder
                st.success("Profiles extracted (placeholder)")
            else:
                st.warning("Please enter some profile text")

# Page 2: Project Setup
elif page == "⚙️ Project Setup":
    st.header("⚙️ Project Requirements")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        project_title = st.text_input("Project Title", placeholder="E-commerce Platform Redesign")
        project_desc = st.text_area("Project Description", height=150,
                                   placeholder="We need to redesign our e-commerce platform with new features...")
    
    with col2:
        team_size = st.slider("Team Size", 2, 10, 4)
        duration = st.selectbox("Duration", ["2 weeks", "1 month", "3 months", "6 months+"])
    
    st.markdown("### Required Roles & Skills")
    
    role_col, skill_col = st.columns(2)
    
    with role_col:
        required_roles = st.multiselect(
            "Required Roles",
            ["Frontend Developer", "Backend Developer", "UX Designer", "Data Scientist", 
             "Project Manager", "QA Engineer", "DevOps", "ML Engineer"],
            default=["Frontend Developer", "Backend Developer"]
        )
    
    with skill_col:
        required_skills = st.multiselect(
            "Required Skills",
            ["Python", "JavaScript", "React", "Node.js", "Figma", "SQL", "Docker", 
             "AWS", "ML", "Data Analysis", "UI/UX", "Agile"],
            default=["Python", "React", "SQL"]
        )
    
    if st.button("🚀 Generate Team Recommendations", type="primary"):
        if not project_desc:
            st.warning("Please enter a project description")
        else:
            # Create project object
            project = {
                "title": project_title,
                "description": project_desc,
                "required_roles": required_roles,
                "required_skills": required_skills,
                "team_size": team_size,
                "duration": duration
            }
            st.session_state.project = project
            
            # Show processing steps
            with st.status("Building optimal teams...", expanded=True) as status:
                st.write("📊 Step 1: Generating candidate embeddings (M1)")
                st.write("🎯 Step 2: Generating project embedding (M2)")
                st.write("👥 Step 3: Running team construction algorithm (M3)")
                st.write("💬 Step 4: Generating explanations (M5)")
                
                # Simulate processing
                import time
                time.sleep(2)
                
                # Placeholder teams
                st.session_state.teams = [
                    [{"name": "Alice", "role": "Frontend", "skills": ["React", "JavaScript"]},
                     {"name": "Bob", "role": "Backend", "skills": ["Python", "Node.js"]}],
                    [{"name": "Charlie", "role": "Full Stack", "skills": ["React", "Python"]},
                     {"name": "Diana", "role": "UX", "skills": ["Figma", "UI/UX"]}]
                ]
                
                st.session_state.explanations = [
                    "Team 1 has strong frontend-backend separation with complementary skills.",
                    "Team 2 features versatile full-stack developers with design expertise."
                ]
                
                status.update(label="✅ Teams generated!", state="complete")
            
            st.success(f"Generated {len(st.session_state.teams)} team configurations")
            st.rerun()

# Page 3: Build Teams (Algorithm Visualization)
elif page == "👥 Build Teams":
    st.header("👥 Team Construction Process")
    
    if not st.session_state.project:
        st.warning("Please set up a project first on the Project Setup page")
        st.stop()
    
    st.markdown("### Algorithm Visualization")
    
    # Placeholder for M3's algorithm visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Candidate-Project Fit", "0.82", "+0.12 vs random")
    
    with col2:
        st.metric("Team Diversity", "0.76", "+0.18 vs random")
    
    with col3:
        st.metric("Skill Coverage", "92%", "+24% vs random")
    
    # Algorithm steps visualization
    st.markdown("#### Greedy Algorithm Steps")
    
    steps = [
        "1. Start with candidate with highest project fit",
        "2. Add candidate that maximizes: α*fit + β*diversity + γ*coverage",
        "3. Repeat until team size reached",
        "4. Output optimized team"
    ]
    
    for step in steps:
        st.markdown(f"<div style='padding: 10px; margin: 5px 0; background: #f0f9ff; border-radius: 5px;'>{step}</div>", 
                   unsafe_allow_html=True)
    
    # Parameters control
    st.markdown("#### Algorithm Parameters")
    
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        alpha = st.slider("α: Fit Weight", 0.0, 1.0, 0.5, 0.1)
    
    with param_col2:
        beta = st.slider("β: Diversity Weight", 0.0, 1.0, 0.3, 0.1)
    
    with param_col3:
        gamma = st.slider("γ: Coverage Weight", 0.0, 1.0, 0.2, 0.1)

# Page 4: Analytics
elif page == "📊 Analytics":
    st.header("📊 Analytics & Insights")
    
    if not st.session_state.profiles:
        st.warning("Please upload profiles first")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Similarity Matrix", "Skill Distribution", "Team Analytics"])
    
    with tab1:
        st.markdown("### Candidate-Candidate Similarity (M1)")
        # Placeholder for M1's similarity matrix
        st.info("This will display the similarity matrix from M1's module")
        
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
            title="Similarity Matrix (Sample)",
            xaxis_title="Candidate Index",
            yaxis_title="Candidate Index",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Skill Distribution")
        # Placeholder for skill distribution
        skills_data = {"Python": 15, "JavaScript": 12, "React": 10, "SQL": 8, "Figma": 6, "Docker": 5}
        
        fig = px.bar(
            x=list(skills_data.keys()),
            y=list(skills_data.values()),
            title="Skill Frequency in Profiles",
            labels={"x": "Skill", "y": "Count"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Team Composition Analysis")
        # Placeholder for team analytics
        st.info("Team analytics will appear here after team generation")

# Page 5: Results
elif page == "📋 Results":
    st.header("📋 Team Recommendations")
    
    if not st.session_state.teams:
        st.warning("No teams generated yet. Please generate teams on the Project Setup page.")
        st.stop()
    
    st.markdown(f"### Project: **{st.session_state.project.get('title', 'Untitled Project')}**")
    st.markdown(f"*{st.session_state.project.get('description', '')}*")
    
    # Team selection
    selected_team = st.selectbox(
        "Select Team to View",
        [f"Team {i+1}" for i in range(len(st.session_state.teams))],
        index=0
    )
    
    team_idx = int(selected_team.split(" ")[1]) - 1
    
    if team_idx < len(st.session_state.teams):
        team = st.session_state.teams[team_idx]
        explanation = st.session_state.explanations[team_idx] if team_idx < len(st.session_state.explanations) else ""
        
        # Team card
        st.markdown("<div class='team-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("#### 👥 Team Members")
            for member in team:
                st.markdown(f"**{member.get('name', 'Unknown')}** - {member.get('role', 'No role')}")
                
                # Skills badges
                skills = member.get('skills', [])
                if skills:
                    skill_html = " ".join([f"<span class='skill-badge'>{skill}</span>" for skill in skills[:5]])
                    st.markdown(skill_html, unsafe_allow_html=True)
                st.markdown("---")
        
        with col2:
            st.markdown("#### 📈 Metrics")
            st.metric("Project Fit", "0.85")
            st.metric("Diversity", "0.78")
            st.metric("Skill Coverage", "95%")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Explanation
        st.markdown("#### 💬 AI Explanation")
        st.info(explanation if explanation else "Explanation will be generated by M5's module")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Select This Team", type="primary"):
                st.success("Team selected! 🎉")
        
        with col2:
            if st.button("🔄 Regenerate Team"):
                st.info("Regenerating team...")
        
        with col3:
            if st.button("📥 Export as JSON"):
                # Export functionality
                import io
                buffer = io.StringIO()
                json.dump(team, buffer, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=buffer.getvalue(),
                    file_name=f"team_{team_idx+1}.json",
                    mime="application/json"
                )
    
    # All teams overview
    st.markdown("---")
    st.markdown("### All Generated Teams")
    
    for i, team in enumerate(st.session_state.teams):
        with st.expander(f"Team {i+1} - Click to expand"):
            cols = st.columns(min(4, len(team)))
            for idx, member in enumerate(team):
                col_idx = idx % 4
                with cols[col_idx]:
                    st.markdown(f"**{member.get('name', 'Member')}**")
                    st.caption(member.get('role', 'Role'))
                    
                    skills = member.get('skills', [])
                    if skills:
                        st.write(", ".join(skills[:3]))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "SkillSync Team Formation System • Group 45 • TU Wien • 2025</div>",
    unsafe_allow_html=True
)
