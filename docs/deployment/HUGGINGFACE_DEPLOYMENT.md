# Hugging Face Deployment Guide

## Overview

This project deploys to Hugging Face Spaces using a **unified Docker container** that runs both the FastAPI backend and Streamlit frontend together.

### Architecture
- **Deployment**: Single Hugging Face Space
- **UI**: Streamlit on port 7860 (Hugging Face default)
- **API**: FastAPI on port 8001 (internal)
- **Database**: SQLite (in-container)
- **Process Manager**: Supervisor orchestrates both services

---

## Quick Start

### 1. Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Configure:
   - **Name**: `hr-attrition-platform`
   - **License**: Apache 2.0
   - **SDK**: Docker
   - **Hardware**: CPU basic (upgrade if needed)
   - **Visibility**: Public or Private

### 2. Prepare Repository

The project already includes `Dockerfile.huggingface` which is the unified deployment container.

**Files that will be deployed:**
```
hr-attrition-platform/
├── docker/
│   └── Dockerfile.huggingface    # Main deployment file
├── pyproject.toml            # Dependencies
├── poetry.lock               # Locked versions
├── api/                      # FastAPI backend
├── core/                     # Business logic
├── database/                 # Database layer
├── ui/                       # Streamlit frontend
├── outputs/                  # ML model
├── data/                     # Sample data
├── scripts/                  # Utilities
└── .streamlit/              # UI config
```

### 3. Deploy to Hugging Face

#### Option A: Push from Local Repository

```bash
# Clone your repository
cd /path/to/hr-attrition-rate

# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/hr-attrition-platform

# Push to Hugging Face (main branch)
git push hf main
```

#### Option B: Create README.md with Metadata

Create a `README.md` in your Hugging Face Space with metadata header:

```markdown
---
title: HR Attrition Prediction Platform
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# HR Attrition Prediction Platform

AI-powered employee attrition prediction with interactive dashboard and API.

## Features

- 📊 Interactive Streamlit Dashboard
- 🔮 Real-time Attrition Predictions
- 📈 SHAP Explainability Visualizations
- 📥 Excel Export Functionality
- 🎯 Risk Categorization (Low/Medium/High)
- 🚀 FastAPI Backend with Documentation

## How to Use

1. Access the dashboard at the Space URL
2. Upload employee data (CSV or Excel)
3. Click "Predict Attrition" to see results
4. Download predictions as Excel

## Architecture

This Space runs both services in one container:
- **Streamlit UI**: Port 7860 (main access point)
- **FastAPI Backend**: Port 8001 (internal)
- **Supervisor**: Manages both processes

## API Access

The API is available internally at `http://localhost:8001` for the UI to use.

For programmatic access, you can also hit the endpoints directly if needed.

## Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 33 engineered features
- **Coverage**: 76% test coverage

## License

Apache 2.0
```

### 4. Dockerfile Overview

The `docker/Dockerfile.huggingface` includes:

```dockerfile
FROM python:3.13-slim

# Install system dependencies + Poetry + Supervisor
RUN apt-get update && apt-get install -y \
    build-essential curl supervisor

# Install Python dependencies via Poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev

# Copy application code
COPY api/ core/ database/ ui/ outputs/ data/ scripts/ .streamlit/ ./

# Configure Supervisor to run both API and UI
# FastAPI on port 8001, Streamlit on port 7860

# Initialize database on startup
CMD poetry run python -m database.init_db && supervisord
```

### 5. Environment Variables

Set these in your Space settings if needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///db_data/hr_attrition.db` | Database connection string |
| `API_BASE_URL` | `http://localhost:8001` | API URL for UI to connect |
| `DISABLE_DB` | `false` | Disable database features |

### 6. Verify Deployment

Once your Space builds (5-10 minutes):

1. **Check Space Status**: Should show "Running"
2. **Access Dashboard**: Click on your Space URL
3. **Test Upload**: Upload sample data file
4. **Verify Predictions**: Run prediction and check results
5. **Check Logs**: View logs in Space settings if issues occur

---

## Advanced Configuration

### Upgrading Hardware

If you experience performance issues:

1. Go to Space **Settings**
2. Select **"Hardware"**
3. Options:
   - **CPU Basic**: Free (default)
   - **CPU Upgrade**: ~$0.01/hour
   - **GPU T4**: ~$0.60/hour (for very large models)

### Custom Domain

1. Go to Space **Settings**
2. Select **"Custom domain"**
3. Follow instructions to configure DNS

### Private Spaces

1. Set visibility to **Private** during creation
2. Control access via Space settings
3. Share with specific users or teams

---

## Monitoring & Logs

### View Logs

In your Space:
1. Click **"Logs"** tab
2. Monitor both Supervisor, FastAPI, and Streamlit logs
3. Filter by service if needed

### Health Checks

The Dockerfile includes health monitoring. Check:
- Supervisor process status
- FastAPI health endpoint
- Streamlit UI responsiveness

### Common Issues

**Space Won't Start:**
- Check logs for errors
- Verify model file exists in `outputs/`
- Ensure Poetry dependencies install correctly

**UI Can't Connect to API:**
- Verify `API_BASE_URL=http://localhost:8001`
- Check Supervisor config in Dockerfile
- Ensure both services are running

**Model File Too Large:**
- Model is ~30MB, should work fine
- If needed, use Git LFS:
  ```bash
  git lfs install
  git lfs track "outputs/*.pkl"
  git add .gitattributes
  git commit -m "Add LFS tracking"
  ```

---

## GitHub Actions Integration

The project includes CI/CD pipeline that builds the Hugging Face Docker image on every push to `main`.

See `.github/workflows/ci-cd.yml`:

```yaml
docker-huggingface:
  name: Build Hugging Face Docker Image
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Build Hugging Face Docker image
    run: |
      docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:${{ github.sha }} .
```

To automatically deploy to Hugging Face on push:

1. Get your HF token: https://huggingface.co/settings/tokens
2. Add to GitHub Secrets as `HF_TOKEN`
3. Update workflow:

```yaml
- name: Push to Hugging Face
  env:
    HF_TOKEN: ${{ secrets.HF_TOKEN }}
  run: |
    git remote add hf https://YOUR-USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR-USERNAME/hr-attrition-platform
    git push hf main --force
```

---

## Cost Estimate

- **Free Tier**: CPU Basic (FREE) - suitable for demos and small-scale usage
- **Paid Tiers**: 
  - CPU Upgrade: ~$7/month (always on)
  - GPU T4: ~$432/month (always on)
  - Consider sleep mode to reduce costs

---

## Local Testing

Before deploying, test the Docker container locally:

```bash
# Build the image
docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:local .

# Run the container
docker run -d -p 7860:7860 --name test-hf hr-attrition-hf:local

# Wait for services to start
sleep 30

# Test the UI
curl http://localhost:7860

# Test the API (internal)
docker exec test-hf curl http://localhost:8001/health

# View logs
docker logs test-hf

# Stop and remove
docker stop test-hf && docker rm test-hf
```

---

## Alternative: Docker Compose for Local Development

For local development with PostgreSQL instead of SQLite:

```bash
# Use the existing docker-compose.yml
docker-compose up -d

# Access:
# - API: http://localhost:8001
# - UI: http://localhost:8501
# - Database: PostgreSQL on port 5432
```

This uses separate Dockerfiles:
- `docker/Dockerfile.api` - FastAPI service
- `docker/Dockerfile.streamlit` - Streamlit service
- `docker/Dockerfile.database` - Database initialization

---

## Next Steps

1. ✅ Review this guide
2. ⬜ Create Hugging Face Space
3. ⬜ Test Docker build locally first
4. ⬜ Push code to Hugging Face
5. ⬜ Verify deployment works
6. ⬜ Configure environment variables if needed
7. ⬜ Set up GitHub Actions for auto-deploy (optional)
8. ⬜ Monitor logs and performance
9. ⬜ Upgrade hardware if needed

---

## Support

For issues:
- Check Space logs first
- Review docker/Dockerfile.huggingface configuration
- Test locally with Docker
- Open issue on [GitHub](https://github.com/shah-data-scientist/HR-ATTRITION-RATE)

---

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker SDK Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [Git LFS Documentation](https://git-lfs.github.com/)
- [Supervisor Documentation](http://supervisord.org/)
