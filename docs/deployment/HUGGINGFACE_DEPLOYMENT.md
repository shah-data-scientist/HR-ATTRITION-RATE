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

The `docker/Dockerfile.huggingface` uses a multi-stage build for optimal image size:

**Stage 1: Builder**
```dockerfile
FROM python:3.13-slim AS builder

# Install Poetry and build dependencies
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev
```

**Stage 2: Runtime**
```dockerfile
FROM python:3.13-slim

# Install Supervisor (process manager)
RUN apt-get update && apt-get install -y supervisor

# Copy virtualenv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY api/ core/ database/ ui/ outputs/ data/ scripts/ .streamlit/ ./

# Configure Supervisor to run both services
# - FastAPI on port 8001 (internal API)
# - Streamlit on port 7860 (Hugging Face default)

# Expose port 7860 for Hugging Face
EXPOSE 7860

# Initialize database and start both services
CMD python -m database.init_db && supervisord
```

**Key Features:**
- **Multi-stage build**: Reduces final image size (~500MB vs ~1GB)
- **SQLite database**: No external database needed for demos
- **Supervisor**: Manages both FastAPI and Streamlit processes
- **Health checks**: Monitors service status
- **Demo credentials**: Pre-configured API keys for easy testing

**Build locally to test:**
```bash
# Build the image
docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:local .

# Run the container
docker run -d -p 7860:7860 --name test-hf hr-attrition-hf:local

# Wait for services to start (30-40 seconds)
Start-Sleep -Seconds 40

# Test the UI
curl http://localhost:7860

# Test the API (from inside container)
docker exec test-hf curl http://localhost:8001/health

# View logs
docker logs test-hf

# Stop and cleanup
docker stop test-hf
docker rm test-hf
```

### 5. Environment Variables

The Dockerfile includes demo environment variables. For production, you can override these in your Hugging Face Space settings:

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `DATABASE_URL` | `sqlite:///db_data/hr_attrition.db` | Database connection string | No - SQLite works for demos |
| `API_BASE_URL` | `http://localhost:8001` | Internal API URL for UI | No - already configured |
| `API_KEY` | `demo_huggingface_api_key` | API authentication key | No - demo key works |
| `SECRET_KEY` | `demo_secret_key_for_huggingface...` | Flask/session secret | No - demo key works |
| `DISABLE_DB` | `false` | Disable database features | No |

**When to override:**
- **Production deployment**: Change `API_KEY` and `SECRET_KEY` to secure values
- **PostgreSQL**: Change `DATABASE_URL` to external PostgreSQL instance
- **Custom configuration**: Add additional environment variables as needed

**How to set in Hugging Face Space:**
1. Go to your Space **Settings**
2. Scroll to **Repository secrets**
3. Add variables:
   ```
   API_KEY=your-secure-api-key-here
   SECRET_KEY=your-secure-secret-key-min-32-chars-long
   ```
4. Restart your Space for changes to take effect

**Generate secure keys:**
```powershell
# API Key (32 characters)
[System.Web.Security.Membership]::GeneratePassword(32,8)

# Or using OpenSSL (if installed)
openssl rand -hex 32
```

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
- **Check logs**: Click **Logs** tab in your Space
- **Model file missing**: Ensure `outputs/` directory with model file is committed
  ```powershell
  # Check if model exists locally
  Test-Path "outputs\*.pkl"
  
  # If missing, train model first (see README.md)
  # Then commit and push
  git add outputs/
  git commit -m "Add trained model"
  git push hf main
  ```
- **Poetry install fails**: Check `pyproject.toml` and `poetry.lock` are in sync
  ```powershell
  # Regenerate lock file if needed
  poetry lock --no-update
  git add poetry.lock
  git commit -m "Update poetry.lock"
  git push hf main
  ```
- **Out of memory**: Upgrade hardware (see Advanced Configuration section)

**UI Can't Connect to API:**
- **Check Supervisor status**: View logs, should show both services RUNNING
- **Verify API_BASE_URL**: Should be `http://localhost:8001` (container internal)
- **Check port configuration**: Streamlit on 7860, FastAPI on 8001
- **Supervisor not starting services**: Check `/etc/supervisor/conf.d/supervisord.conf` in Dockerfile

**Model File Too Large:**
- Current model is ~30MB, works fine with Git
- For models >50MB, use **Git LFS**:
  ```powershell
  # Install Git LFS
  git lfs install
  
  # Track large files
  git lfs track "outputs/*.pkl"
  git lfs track "outputs/*.joblib"
  
  # Commit tracking config
  git add .gitattributes
  git commit -m "Add LFS tracking for model files"
  
  # Add and commit model
  git add outputs/
  git commit -m "Add model with LFS"
  
  # Push to Hugging Face (LFS will handle large files)
  git push hf main
  ```

**Database Errors:**
- **SQLite locked**: Expected with multiple workers, errors are handled gracefully
- **Table doesn't exist**: Database init may have failed, check logs for errors
- **Switch to PostgreSQL**: For production, use external PostgreSQL:
  ```yaml
  # In Space settings, set:
  DATABASE_URL=postgresql://user:password@external-host:5432/dbname
  ```

**Build Timeout:**
- Hugging Face has 60-minute build timeout
- If exceeded, optimize Dockerfile:
  - Use lighter base image
  - Reduce dependencies
  - Pre-build and push to Docker Hub

---

## GitHub Actions Integration

The project includes a CI/CD pipeline that validates the Hugging Face Docker image builds correctly on every push to `main`.

**Note:** Docker push to GitHub Container Registry is **disabled** to avoid organization package creation errors. The workflow only builds and validates the image.

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
      
  - name: Test image can start
    run: |
      docker run -d --name test-hf hr-attrition-hf:${{ github.sha }}
      sleep 30
      docker logs test-hf
      docker stop test-hf
```

### Automated Deployment to Hugging Face

To automatically deploy to Hugging Face on push:

1. **Get your HF token**: https://huggingface.co/settings/tokens
   - Create token with **Write** access
   - Token format: `hf_...`

2. **Add to GitHub Secrets**:
   - Go to repository **Settings** → **Secrets and variables** → **Actions**
   - Click **New repository secret**
   - Name: `HF_TOKEN`
   - Value: Your token from step 1

3. **Update workflow** to add deployment step:

```yaml
docker-huggingface:
  name: Build and Deploy Hugging Face Image
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
  
  steps:
  - uses: actions/checkout@v4
  
  - name: Build Hugging Face Docker image
    run: |
      docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:${{ github.sha }} .
  
  - name: Test image
    run: |
      docker run -d --name test-hf hr-attrition-hf:${{ github.sha }}
      sleep 30
      docker exec test-hf curl -f http://localhost:8001/health || exit 1
      docker stop test-hf && docker rm test-hf
  
  - name: Push to Hugging Face Space
    env:
      HF_TOKEN: ${{ secrets.HF_TOKEN }}
      HF_USERNAME: YOUR-USERNAME
      HF_SPACE: hr-attrition-platform
    run: |
      # Configure git
      git config --global user.email "github-actions[bot]@users.noreply.github.com"
      git config --global user.name "GitHub Actions"
      
      # Add Hugging Face remote
      git remote add hf https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE || true
      
      # Push to Hugging Face
      git push hf main --force
```

**Important Notes:**
- Replace `YOUR-USERNAME` with your Hugging Face username
- Replace `hr-attrition-platform` with your Space name
- The workflow will push **entire repository** to Hugging Face Space
- Hugging Face will automatically rebuild using `docker/Dockerfile.huggingface`
- First deployment may take 5-10 minutes

### Alternative: Manual Deployment

If you prefer manual deployment:

```bash
# Add remote once
git remote add hf https://YOUR-USERNAME:YOUR-HF-TOKEN@huggingface.co/spaces/YOUR-USERNAME/hr-attrition-platform

# Deploy whenever ready
git push hf main

# Or force push to overwrite
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

### Automated Testing (Recommended)

We provide a PowerShell script to automate the local testing process, including building the image, running the container with appropriate environment variables, and checking health endpoints.

```powershell
# Run the test script
.\scripts\test-hf-local.ps1
```

This script will:
1. Build the Docker image (`hr-attrition-hf:test`)
2. Run the container with an in-memory SQLite database
3. Expose the UI on port 7860 and API on port 8001
4. Wait for services to start and verify health
5. Show access URLs and credentials

### Manual Testing

If you prefer to run commands manually:

```powershell
# Build the image (takes 3-5 minutes)
docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:local .

# Run the container
# Note: We expose port 8001 for API access from host
docker run -d -p 7860:7860 -p 8001:8001 --name test-hf hr-attrition-hf:local

# Wait for services to start (service startup)
Start-Sleep -Seconds 40

# Test the UI (should return HTML)
curl http://localhost:7860

# Test the API health endpoint (from host)
curl http://localhost:8001/health

# Check both services are running
docker exec test-hf supervisorctl status

# View real-time logs
docker logs -f test-hf

# Stop following logs with Ctrl+C, then cleanup
docker stop test-hf
docker rm test-hf
```

**Expected output:**
```bash
# supervisorctl status should show:
fastapi                          RUNNING   pid 123, uptime 0:00:30
streamlit                        RUNNING   pid 124, uptime 0:00:30

# Health endpoint should return:
{"status":"healthy","database":"connected","model":"loaded"}
```

**Troubleshooting local test:**

| Issue | Cause | Solution |
|-------|-------|----------|
| Container exits immediately | Database init failed | `docker logs test-hf` to see errors |
| UI not accessible | Services still starting | Wait 60 seconds, Streamlit is slower to start |
| API returns 500 | Model file missing | Ensure `outputs/` directory exists with model |
| "Connection refused" | Port already in use | Stop other services on 7860, or use `-p 8080:7860` |

**Test with sample data:**
1. Access http://localhost:7860 in browser
2. Upload one of the sample CSV files from `data/` folder
3. Click "Predict Attrition"
4. Verify predictions display correctly
5. Test Excel export functionality

---

## Alternative: Docker Compose for Local Development

For local development with full PostgreSQL database instead of SQLite:

```powershell
# Use the existing docker-compose.yml
docker-compose up -d

# Wait for all services to start
Start-Sleep -Seconds 60

# Check service status
docker-compose ps

# Access services:
# - API: http://localhost:8001
# - API Docs: http://localhost:8001/docs
# - UI: http://localhost:8501
# - Database: PostgreSQL on port 5432

# View logs
docker-compose logs -f fastapi_app
docker-compose logs -f streamlit_app

# Stop services
docker-compose down
```

**Docker Compose Services:**

| Service | Dockerfile | Port | Purpose |
|---------|-----------|------|---------|
| `db` | postgres:16-alpine | 5432 (internal) | PostgreSQL database |
| `db_init` | docker/Dockerfile.database | - | Initialize database schema |
| `fastapi_app` | docker/Dockerfile.api | 8001 | FastAPI backend |
| `streamlit_app` | docker/Dockerfile.streamlit | 8501 | Streamlit UI |
| `worker` | docker/Dockerfile.api | - | Background job processor |

**Key Differences from Hugging Face:**

| Feature | Hugging Face (Production) | Docker Compose (Development) |
|---------|---------------------------|------------------------------|
| Services | Combined (Supervisor) | Separate containers |
| Database | SQLite (single file) | PostgreSQL (full DB server) |
| Port | 7860 (Streamlit only) | 8001 (API) + 8501 (UI) |
| API Access | Internal only | Exposed on localhost |
| Background Jobs | Not included | Worker service included |
| Best for | Demos, deployments | Development, testing |

**Environment Configuration:**

Create a `.env` file:
```bash
# Database
POSTGRES_DB=hr_attrition_db
POSTGRES_USER=user
POSTGRES_PASSWORD=secure_password

# API
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_min_32_chars

# Development
DISABLE_DB=false
```

**Advantages of Docker Compose:**
- ✅ Separate services for easier debugging
- ✅ PostgreSQL for production-like environment
- ✅ Worker service for background jobs
- ✅ API accessible directly for testing
- ✅ Hot reload with volume mounts (can be configured)

**Switching between environments:**
```powershell
# Development: Docker Compose
docker-compose up -d
# Access: http://localhost:8501

# Production test: Hugging Face Dockerfile
docker build -f docker/Dockerfile.huggingface -t hf-test .
docker run -p 7860:7860 hf-test
# Access: http://localhost:7860
```

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
