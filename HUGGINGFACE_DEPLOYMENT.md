# Hugging Face Deployment Guide

## Overview
This project will be deployed as two separate Hugging Face Spaces:
1. **API Space** - FastAPI backend (port 7860)
2. **UI Space** - Streamlit frontend (connects to API)

---

## 1. API Deployment (FastAPI Space)

### Create Space
- Go to https://huggingface.co/spaces
- Click "Create new Space"
- Name: `hr-attrition-api`
- License: Apache 2.0
- SDK: **Docker**
- Hardware: CPU basic (or upgrade if needed)

### Files Required

**Root directory structure:**
```
hr-attrition-api/
├── Dockerfile
├── requirements.txt
├── README.md
├── api/
├── core/
├── database/
├── outputs/
└── data/
```

### Dockerfile for API
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY core/ ./core/
COPY database/ ./database/
COPY outputs/ ./outputs/
COPY data/ ./data/

# Expose Hugging Face default port
EXPOSE 7860

# Run API on port 7860 (Hugging Face requirement)
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### requirements.txt for API
```txt
fastapi==0.115.6
uvicorn==0.34.0
pydantic==2.10.3
pandas==2.2.3
numpy==2.2.1
scikit-learn==1.6.0
xgboost==2.1.3
shap==0.46.0
sqlalchemy==2.0.36
openpyxl==3.1.5
python-multipart==0.0.20
```

### README.md for API Space
```markdown
---
title: HR Attrition Prediction API
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# HR Attrition Prediction API

FastAPI backend for HR employee attrition prediction with SHAP explainability.

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /predict/attrition` - Predict attrition for employees
- `POST /predict/attrition/excel` - Download predictions as Excel
- `POST /predict/shap` - Generate SHAP explanation images

## Usage

```python
import requests

url = "https://YOUR-USERNAME-hr-attrition-api.hf.space/predict/attrition"
data = {
    "employees": [
        {
            "id_employee": "EMP001",
            "age": 35,
            "genre": "M",
            # ... other fields
        }
    ]
}
response = requests.post(url, json=data)
print(response.json())
```

## Model Info
- Algorithm: XGBoost Classifier
- Features: 33 engineered features
- Coverage: 76% test coverage
```

---

## 2. UI Deployment (Streamlit Space)

### Create Space
- Go to https://huggingface.co/spaces
- Click "Create new Space"
- Name: `hr-attrition-ui`
- License: Apache 2.0
- SDK: **Streamlit**
- Hardware: CPU basic

### Files Required

**Root directory structure:**
```
hr-attrition-ui/
├── app.py (renamed from ui/app.py)
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml
```

### app.py (Main Streamlit App)
Copy `ui/app.py` and modify the API URL:

```python
# At the top of the file, modify API_BASE_URL
API_BASE_URL = "https://YOUR-USERNAME-hr-attrition-api.hf.space"
```

### requirements.txt for UI
```txt
streamlit==1.41.1
pandas==2.2.3
requests==2.32.3
plotly==5.24.1
openpyxl==3.1.5
```

### .streamlit/config.toml
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 7860
enableCORS = false
enableXsrfProtection = false
```

### README.md for UI Space
```markdown
---
title: HR Attrition Prediction Dashboard
emoji: 👥
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
---

# HR Attrition Prediction Dashboard

Interactive Streamlit dashboard for predicting employee attrition risk.

## Features

- 📊 Upload employee data (CSV/Excel)
- 🔮 Real-time attrition predictions
- 📈 SHAP explainability visualizations
- 📥 Download results as Excel
- 🎯 Risk categorization (Low/Medium/High)

## Data Format

Required columns:
- id_employee, age, genre, revenu_mensuel
- satisfaction_employee_*, note_evaluation_*
- departement, poste, statut_marital
- And more... (see sample data in app)

## Connected API

This UI connects to: `hr-attrition-api` Space
```

---

## 3. Deployment Steps

### Step 1: Prepare API Space

```bash
# Create new git repo for API
cd /path/to/new/folder
git init
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/hr-attrition-api

# Copy files
cp -r api/ core/ database/ outputs/ data/ .
cp Dockerfile.api Dockerfile
# Create requirements.txt (see above)
# Create README.md (see above)

# Commit and push
git add .
git commit -m "Initial API deployment"
git push origin main
```

### Step 2: Prepare UI Space

```bash
# Create new git repo for UI
cd /path/to/new/folder
git init
git remote add origin https://huggingface.co/spaces/YOUR-USERNAME/hr-attrition-ui

# Copy and rename files
cp ui/app.py app.py
mkdir .streamlit
# Create config.toml (see above)
# Create requirements.txt (see above)
# Create README.md (see above)

# IMPORTANT: Update API URL in app.py
# Change API_BASE_URL to your API Space URL

# Commit and push
git add .
git commit -m "Initial UI deployment"
git push origin main
```

### Step 3: Configure Spaces

1. **API Space Settings:**
   - Go to Space settings
   - Add secrets if needed (database credentials, etc.)
   - Set hardware (CPU basic should work)
   - Enable analytics if desired

2. **UI Space Settings:**
   - Go to Space settings
   - Add secret: `API_URL` = your API space URL
   - Update app.py to read from environment variable:
     ```python
     import os
     API_BASE_URL = os.getenv("API_URL", "https://YOUR-USERNAME-hr-attrition-api.hf.space")
     ```

### Step 4: Test Deployment

1. Wait for spaces to build (5-10 minutes)
2. Test API: `https://YOUR-USERNAME-hr-attrition-api.hf.space/health`
3. Test UI: `https://YOUR-USERNAME-hr-attrition-ui.hf.space`
4. Test full workflow: Upload data in UI → See predictions

---

## 4. Environment Variables / Secrets

### API Space Secrets (if needed)
- `DATABASE_URL` - If using external PostgreSQL
- `MODEL_PATH` - Custom model location
- `DISABLE_DB` - Set to "true" for stateless mode

### UI Space Secrets
- `API_URL` - Full URL to API Space (recommended)

---

## 5. Troubleshooting

### API Issues
- Check logs in Space page
- Verify model file exists in `outputs/`
- Ensure port 7860 is used
- Check memory usage (upgrade hardware if needed)

### UI Issues
- Verify API_URL is correct
- Check CORS settings (should be disabled for Hugging Face)
- Test API endpoint separately
- Check Streamlit logs

### Common Fixes
```bash
# If model file too large (>50MB)
# Use Git LFS
git lfs install
git lfs track "outputs/*.pkl"
git add .gitattributes
git commit -m "Add LFS tracking"
```

---

## 6. GitHub Actions Integration

Add to `.github/workflows/deploy-hf.yml`:

```yaml
name: Deploy to Hugging Face

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy-api:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Push to HF API Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        git remote add hf-api https://YOUR-USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR-USERNAME/hr-attrition-api
        git push hf-api main --force

  deploy-ui:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Push to HF UI Space
      env:
        HF_TOKEN: ${{ secrets.HF_TOKEN }}
      run: |
        git remote add hf-ui https://YOUR-USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR-USERNAME/hr-attrition-ui
        git subtree push --prefix ui hf-ui main
```

---

## 7. Cost Estimate

- **Free Tier**: Both spaces on CPU basic (FREE)
- **Upgraded**: 
  - CPU Upgrade: ~$0.01/hour per space
  - GPU T4: ~$0.60/hour (if needed for large models)

---

## 8. Next Steps

1. ✅ Review this document
2. ⬜ Create both Hugging Face Spaces
3. ⬜ Prepare files (Dockerfile, requirements, README)
4. ⬜ Update API URL in UI code
5. ⬜ Test locally with Docker first
6. ⬜ Push to Hugging Face
7. ⬜ Test deployed version
8. ⬜ Set up GitHub Actions (optional)

---

## Notes
- Model file (`employee_attrition_pipeline.pkl`) is ~30MB - should upload fine
- Database will be SQLite (local) unless you configure external PostgreSQL
- SHAP images will be generated in-memory (no persistent storage needed)
- Consider adding rate limiting for production use
