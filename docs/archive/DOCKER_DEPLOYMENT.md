# Docker Deployment Guide

This project uses a single `docker-compose.yml` file with three deployment profiles.

## Quick Start

### Local Development (Default)
```bash
# Start all services for local development
docker-compose --profile local up -d

# Or rebuild containers
docker-compose --profile local up -d --build

# Access:
# - Streamlit UI: http://localhost:8581
# - FastAPI: http://localhost:8081
```

### Production Deployment
```bash
# Start all services for production
docker-compose --profile prod up -d

# Access:
# - Streamlit UI: http://localhost:8501
# - FastAPI: http://localhost:8001
```

### Hugging Face Spaces
```bash
# Start single container for Hugging Face
docker-compose --profile huggingface up -d

# Access:
# - Streamlit UI: http://localhost:7860
# - FastAPI: http://localhost:8001
```

## Profile Comparison

| Feature | Local (`local`) | Production (`prod`) | Hugging Face (`huggingface`) |
|---------|----------------|---------------------|------------------------------|
| **Containers** | 4 separate | 5 separate + worker | 1 combined |
| **Database** | PostgreSQL | PostgreSQL | SQLite |
| **Ports** | 8581, 8081 | 8501, 8001 | 7860, 8001 |
| **Use Case** | Development | Production | Cloud deployment |
| **Logging** | Basic | Advanced (10MB limit) | Basic |
| **Resource Limits** | Moderate | High | Moderate |

## Common Operations

### View Logs
```bash
# Local
docker-compose --profile local logs -f streamlit_app

# Production
docker-compose --profile prod logs -f streamlit_app_prod

# Hugging Face
docker-compose --profile huggingface logs -f huggingface
```

### Stop Services
```bash
docker-compose --profile local down      # Local
docker-compose --profile prod down       # Production
docker-compose --profile huggingface down # Hugging Face
```

### Rebuild Specific Service
```bash
# Local
docker-compose --profile local up -d --build streamlit_app

# Production
docker-compose --profile prod up -d --build streamlit_app_prod

# Hugging Face
docker-compose --profile huggingface up -d --build huggingface
```

### Clean Up Everything
```bash
# Stop and remove all containers, networks, and volumes
docker-compose --profile local down -v
docker-compose --profile prod down -v
docker-compose --profile huggingface down -v
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Database (PostgreSQL for local/prod)
POSTGRES_DB=hr_attrition_db
POSTGRES_USER=user
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql://user:your_secure_password@db:5432/hr_attrition_db

# API Security
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_at_least_32_chars

# UI Authentication
UI_ADMIN_USERNAME=admin
UI_ADMIN_PASSWORD=Admin@2025!Secure
UI_USER_USERNAME=analyst
UI_USER_PASSWORD=Analyst@2025!View

# Worker (Production only)
WORKER_POLL_SEC=2
WORKER_STALE_SEC=600
```

## Migrating from Old Files

If you have old docker-compose files, you can now delete them:

```bash
# Backup first (optional)
mkdir docker-compose-backup
mv docker-compose.prod.yml docker-compose-backup/
mv docker-compose-huggingface.yml docker-compose-backup/

# Or delete directly
rm docker-compose.prod.yml
rm docker-compose-huggingface.yml
```

## Troubleshooting

### Port Conflicts
- **Local**: Uses ports 8581 and 8081 to avoid conflicts
- **Production**: Uses standard ports 8501 and 8001
- **Hugging Face**: Uses port 7860 (HF default)

### Database Connection Issues
```bash
# For PostgreSQL (local/prod)
docker-compose --profile local down -v  # Remove old volumes
docker-compose --profile local up -d --build  # Rebuild with fresh DB
```

### Health Check Failures
```bash
# Check logs
docker-compose --profile local logs db
docker-compose --profile local logs fastapi_app

# Restart services
docker-compose --profile local restart
```
