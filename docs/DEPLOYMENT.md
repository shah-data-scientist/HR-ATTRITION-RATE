# Deployment Guide

## Docker Compose (recommended)

The project ships with a single `docker-compose.yml` that has two named profiles.

### Local development profile

Uses non-standard ports to avoid conflicts with other local services.

```bash
docker compose --profile local up -d
```

| Service | Internal port | Host port |
|---------|--------------|-----------|
| PostgreSQL | 5432 | 5432 |
| FastAPI | 8001 | 8081 |
| Streamlit | 8501 | 8581 |

### Production profile

Uses standard ports, adds resource limits, log rotation, and a background worker.

```bash
docker compose --profile prod up -d
```

| Service | Host port |
|---------|-----------|
| FastAPI | 8001 |
| Streamlit | 8501 |

The production profile adds a `worker` container that processes async report jobs.

---

## Environment variables for production

Copy and fill in `.env` before starting:

```bash
# Authentication — use strong random values
API_KEY=<64-char hex string>
SECRET_KEY=<32+ char secret>

# Database
DATABASE_URL=postgresql://user:password@db:5432/hr_attrition_db
POSTGRES_USER=user
POSTGRES_PASSWORD=<strong password>
POSTGRES_DB=hr_attrition_db

# Worker tuning
WORKER_POLL_SEC=2
WORKER_STALE_SEC=600
```

Generate strong keys:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Container architecture

Three custom images are built from `docker/`:

| Dockerfile | Image | Base |
|-----------|-------|------|
| `Dockerfile.api` | FastAPI + worker | python:3.13-slim, multi-stage |
| `Dockerfile.streamlit` | Streamlit UI | python:3.13-slim, multi-stage |
| `Dockerfile.database` | DB init | python:3.13-slim |

All containers run as non-root users (`appuser`, UID 1000/1001).

The worker container reuses the API image with a different CMD:
```yaml
command: ["python", "-m", "scripts.worker"]
```

---

## CI/CD — GitHub Actions

The pipeline at `.github/workflows/ci-cd.yml` runs on every push to `main`:

1. **code-quality** — Black, Mypy, Ruff
2. **security-scan** — Trivy → GitHub Security
3. **test-with-database** — pytest with live PostgreSQL + Codecov
4. **test-authentication** — bcrypt and API key tests
5. **build-docker-images** — builds and pushes to `ghcr.io` (on `main` branch only)

Images are tagged by branch name and commit SHA, and pushed to:
```
ghcr.io/shah-data-scientist/hr-attrition-rate-api:<tag>
ghcr.io/shah-data-scientist/hr-attrition-rate-ui:<tag>
```

Required GitHub secrets: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `API_KEY`.

---

## Health checks

All containers have built-in health checks:

```bash
# API
curl http://localhost:8001/health

# UI
curl http://localhost:8501/_stcore/health

# Database
docker compose ps    # check "healthy" status
```

---

## Resource limits (production profile)

| Service | CPU limit | Memory limit |
|---------|-----------|--------------|
| PostgreSQL | 1.0 | 512M |
| FastAPI | 2.0 | 2G |
| Streamlit | 1.0 | 1G |
| Worker | 2.0 | 2G |

---

## Logs

Production containers use `json-file` logging driver with rotation:
- API / Worker: 10MB per file, 5 files max
- Streamlit: 10MB per file, 3 files max

```bash
docker compose logs -f fastapi_app_prod
docker compose logs -f worker
```
