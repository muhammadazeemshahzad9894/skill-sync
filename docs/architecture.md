# SkillSync Architecture

## System Overview

SkillSync is an AI-powered team formation system that automatically creates balanced, diverse, and complementary teams by analyzing skills, experience, and preferences.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit UI (M4)                   │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────┐
│                  Integration Layer                   │
└─────┬──────────────┬──────────────┬─────────────┬────┘
      │              │              │             │
┌─────▼─────┐  ┌───-─▼─────┐  ┌───-─▼────┐  ┌──-──▼────┐
│   JSON    │  │ Candidate │  │ Project  │  │  Team    │
│Extraction │  │Embeddings │  │Embeddings│  │  Builder │
│   (M5)    │  │   (M1)    │  │   (M2)   │  │   (M3)   │
└─────┬─────┘  └─-───┬─────┘  └───-─┬────┘  └───-─┬────┘
      │              │              │             │
      └──────────────┴──────┬───────┴─────────────┘
                            │
                  ┌─────────▼─────────┐
                  │    Explanations   │
                  │       (M5)        │
                  └───────────────────┘
```

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

All modules communicate through defined interfaces in `shared/interfaces.py`:

1. `extract_profiles_from_raw()` - M5 → All
2. `generate_candidate_embeddings()` - M1 → M3, M4
3. `generate_project_embedding()` - M2 → M3, M4  
4. `build_teams()` - M3 → M4, M5
5. `generate_explanations()` - M5 → M4

## Testing Strategy

1. **Unit Tests**: Each member tests their own module
2. **Integration Tests**: Complete pipeline test
3. **End-to-End Tests**: Full system with sample data

## Deployment

Local development:
```bash
pip install -r requirements.txt
streamlit run src/modules/member4_ui/app.py
```

## Future Extensions

1. Multi-project team assignment
2. Learning from feedback
3. Integration with HR systems
4. Advanced visualization dashboards
