# Development Guide

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- OpenAI API key (for M5)

### Installation
```bash
# Clone repository
git clone https://github.com/akaysaban/skill-sync.git
cd skill-sync

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env
# Edit .env and add your OpenAI API key
```

### Running the Application
```bash
# Start Streamlit UI
streamlit run src/modules/member4_ui/app.py

# Run integration test
python tests/integration_test.py
```

## Module Development

### Each Member's Workspace
Each member works in their own directory:
- `src/modules/member1_candidate_embeddings/` - Shahzad
- `src/modules/member2_project_embeddings/` - Roxana  
- `src/modules/member3_team_builder/` - Ana
- `src/modules/member4_ui/` - Șaban
- `src/modules/member5_json_explainer/` - Noor

### Implementing Your Module

1. **Understand the interface** in `shared/interfaces.py`
2. **Create your main module file** (e.g., `embeddings.py` for M1)
3. **Implement the required functions** from the interface
4. **Create tests** in your module directory
5. **Update requirements.txt** if you need new packages

### Example: M1 Creating Candidate Embeddings

```python
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
```

## Git Workflow

### Branch Strategy
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/memberX-description`: Feature branches

### Daily Workflow
1. Pull latest changes
```bash
git checkout develop
git pull origin develop
```

2. Create feature branch
```bash
git checkout -b feature/m1-candidate-embeddings
```

3. Work on your feature
4. Commit regularly
```bash
git add .
git commit -m "M1: Implemented candidate embedding generation"
```

5. Push to remote
```bash
git push origin feature/m1-candidate-embeddings
```

6. Create Pull Request to `develop`

### Commit Message Convention
Prefix with member identifier:
- `M1:` - Shahzad
- `M2:` - Roxana
- `M3:` - Ana  
- `M4:` - Șaban
- `M5:` - Noor

Examples:
```
M1: Added cosine similarity computation
M4: Fixed UI layout issue
M5: Improved JSON extraction prompts
```

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_member1_embeddings.py

# Run with coverage
pytest --cov=src tests/
```

### Creating Tests
Create test files in `tests/` directory:
- `test_member1_embeddings.py`
- `test_member2_project_embeddings.py`
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
```python
import sys
sys.path.append('path/to/skill-sync/src')
```

### API Key Issues
Store API keys in `.env` file and load with:
```python
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
```

### Performance Issues
- Use FAISS for vector search (not brute force)
- Cache embeddings to disk
- Batch process when possible

## Getting Help

1. Check `docs/` directory first
2. Look at sample code in `data/sample/`
3. Ask in team chat
4. Create GitHub issue for bugs

## Code Quality Guidelines

1. **Documentation**: Docstrings for all functions
2. **Type Hints**: Use Python type annotations
3. **Error Handling**: Use try-except for external calls
4. **Logging**: Use logging module, not print statements
5. **Code Style**: Follow PEP 8, use Black formatter

Example:
```python
def calculate_similarity(vec1: np.array, vec2: np.array) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have same dimension")
    
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```
