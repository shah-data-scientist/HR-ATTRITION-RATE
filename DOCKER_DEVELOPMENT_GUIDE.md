# Docker Development Guide - HR Attrition Rate Project

**Quick Reference Guide for Local Development vs Docker Workflow**

---

## Table of Contents

1. [The Problem We Solved](#the-problem-we-solved)
2. [Understanding the Issue](#understanding-the-issue)
3. [Best Practice Workflow](#best-practice-workflow)
4. [When to Rebuild Docker](#when-to-rebuild-docker)
5. [Quick Commands](#quick-commands)
6. [Helper Scripts](#helper-scripts)
7. [Common Mistakes](#common-mistakes)
8. [Port Management](#port-management)
9. [Troubleshooting](#troubleshooting)

---

## The Problem We Solved

### What Went Wrong

Multiple Streamlit instances were running simultaneously:
- **Local Streamlit** (from previous dev sessions) on port 8501 with old config
- **Docker Streamlit** container also on port 8501 with new config

When accessing `http://localhost:8501`, the browser connected to the **local instance** instead of Docker, showing the old API endpoint even though Docker was correctly configured.

### Root Cause

**Forgot to run `docker-compose down` before starting local development**, causing:
- Port conflicts (8501, 8001)
- Mixed environment variables
- Confusion about which instance was running
- Old configurations persisting

### The Fix

**Always run `docker-compose down` before switching modes!**

---

## Understanding the Issue

### Docker vs Local Development

You have **ONE deployment method at a time**, not two:

| Aspect | Local Dev | Docker |
|--------|-----------|--------|
| **How to start** | `poetry run streamlit run ui/app.py` | `docker-compose up -d` |
| **Ports** | 8502, 8002 (different) | 8501, 8001 (standard) |
| **Environment** | Your PC directly | Isolated containers |
| **Database** | Separate install needed | Containerized PostgreSQL |
| **Use case** | Quick code changes | Full system testing |
| **Where it runs** | Directly on Windows | Inside Linux containers |

### Docker ≠ Production

Docker is **containerization technology** that runs:
- On your local PC (what you're doing now)
- On cloud servers (AWS, Azure, GCP)
- On-premises servers
- Kubernetes clusters

Think of Docker as **portable packaging**, not a production server.

---

## Best Practice Workflow

### Phase 1: Local Development (90% of time)

**Fast iteration for code changes**

```bash
# 1. Clean slate - Stop Docker first!
docker-compose down

# 2. Verify ports are free (optional but recommended)
netstat -ano | findstr ":8501 :8001"

# 3. Start local development
poetry run streamlit run ui/app.py --server.port 8502

# 4. In another terminal, start API (optional)
poetry run uvicorn api.main:app --port 8002 --reload

# 5. Access at:
# http://localhost:8502 (Streamlit)
# http://localhost:8002/docs (API)
```

**Why local dev first?**
- ⚡ Instant feedback - changes appear immediately
- 🐛 Easy debugging - set breakpoints, use debugger
- 🚀 Fast iteration - no rebuild/restart needed
- 💻 Native performance - runs directly on your machine

### Phase 2: Docker Testing (10% of time)

**Full stack testing before commit**

```bash
# 1. Stop local dev
# Press Ctrl+C in all terminals

# 2. Rebuild Docker (IMPORTANT!)
docker-compose build

# 3. Start Docker
docker-compose up -d

# 4. Test at http://localhost:8501

# 5. Check logs if needed
docker-compose logs -f

# 6. When satisfied, stop Docker
docker-compose down
```

**Why Docker before commit?**
- 🔒 Environment consistency - same as production
- 🧪 Full stack testing - DB, API, UI together
- 🐳 Catch Docker-specific issues - permissions, paths
- ✅ Deployment confidence - works locally = works on server

### Phase 3: Commit & Deploy

```bash
# After Docker tests pass
git add .
git commit -m "Add feature X"
git push

# Same Docker setup works on server!
```

---

## When to Rebuild Docker

### Decision Tree

Docker images are **snapshots** of your code at build time. Changes to local files don't automatically update existing images.

#### ✅ MUST Rebuild - Code Changed

```bash
# You edited ANY Python files
# ui/app.py, api/main.py, scripts/worker.py, etc.

docker-compose build
docker-compose up -d
```

#### ✅ MUST Rebuild - Dependencies Changed

```bash
# You added/removed packages
poetry add requests
poetry remove pandas

docker-compose build --no-cache  # Full clean rebuild
docker-compose up -d
```

#### ✅ MUST Rebuild - Dockerfile Changed

```bash
# You edited any Dockerfile
# docker/Dockerfile.api, docker/Dockerfile.streamlit

docker-compose build --no-cache
docker-compose up -d
```

#### ⚠️ Restart Only - Config Changed

```bash
# You ONLY changed .env file
# No code changes

docker-compose down
docker-compose up -d  # No build needed
```

#### ⚠️ Restart Only - docker-compose.yml Environment

```bash
# You changed environment variables in docker-compose.yml
# But NOT code files

docker-compose down
docker-compose up -d  # No build needed
```

---

## Quick Commands

### Local Development

```bash
# Check ports are free
netstat -ano | findstr ":8501 :8001"

# Start Streamlit (local)
poetry run streamlit run ui/app.py --server.port 8502

# Start API (local, with auto-reload)
poetry run uvicorn api.main:app --port 8002 --reload
```

### Docker Commands

```bash
# Clean stop everything
docker-compose down

# Build from scratch
docker-compose build --no-cache

# Incremental build (faster)
docker-compose build

# Start detached
docker-compose up -d

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f streamlit_app
docker-compose logs -f fastapi_app
docker-compose logs -f worker

# Check status
docker ps

# Full restart
docker-compose down && docker-compose build && docker-compose up -d
```

### Switching Between Modes

```bash
# Local → Docker
1. Ctrl+C (stop local servers)
2. docker-compose build
3. docker-compose up -d

# Docker → Local
1. docker-compose down
2. poetry run streamlit run ui/app.py --server.port 8502
```

---

## Helper Scripts

### check-ports.bat

Check if ports are free before starting services.

```batch
@echo off
echo Checking if ports 8501, 8001, 5432 are free...
echo.
netstat -ano | findstr "LISTENING" | findstr ":8501 :8001 :5432"
if %errorlevel% == 0 (
    echo.
    echo WARNING: Some ports are in use!
    echo Run: docker-compose down
    echo Or kill the processes shown above
) else (
    echo All ports are FREE - safe to proceed!
)
```

### dev-local.bat

Start local development with safety checks.

```batch
@echo off
echo ========================================
echo Starting LOCAL DEVELOPMENT MODE
echo ========================================
echo.
echo Stopping Docker containers...
docker-compose down
echo.
echo Checking ports are free...
netstat -ano | findstr ":8501 :8001" && (
    echo WARNING: Ports still in use!
    pause
    exit /b 1
)
echo.
echo Starting Streamlit on port 8502...
poetry run streamlit run ui/app.py --server.port 8502
```

### dev-docker.bat

Start Docker mode with cleanup.

```batch
@echo off
echo ========================================
echo Starting DOCKER MODE
echo ========================================
echo.
echo Stopping any local processes...
echo (Press Ctrl+C in local dev terminals if running)
timeout /t 3
echo.
echo Building and starting Docker...
docker-compose up -d
echo.
echo Waiting for containers to be healthy...
timeout /t 10
docker ps
echo.
echo ========================================
echo Docker running at http://localhost:8501
echo ========================================
```

### rebuild-docker.bat

Smart rebuild after local development.

```batch
@echo off
echo ========================================
echo REBUILDING DOCKER AFTER LOCAL DEV
echo ========================================
echo.

echo Step 1: Stopping containers...
docker-compose down

echo.
echo Step 2: Checking what changed...
git status --short

echo.
echo Step 3: Rebuilding images...
set /p CLEAN="Clean rebuild (slower)? (y/N): "
if /i "%CLEAN%"=="y" (
    echo Running CLEAN rebuild...
    docker-compose build --no-cache
) else (
    echo Running INCREMENTAL rebuild...
    docker-compose build
)

echo.
echo Step 4: Starting containers...
docker-compose up -d

echo.
echo Step 5: Checking status...
timeout /t 5 >nul
docker ps

echo.
echo ========================================
echo Rebuild complete!
echo Access at: http://localhost:8501
echo Logs: docker-compose logs -f
echo ========================================
```

---

## Common Mistakes

### ❌ Mistake 1: Not Running `docker-compose down`

**The Most Common Error!**

```bash
# BAD
poetry run streamlit run ui/app.py --server.port 8502
# ... work ...
# Ctrl+C
docker-compose up -d  # ❌ Port conflicts! Old instance still running!
```

```bash
# GOOD
docker-compose down   # ✅ Clean slate first!
poetry run streamlit run ui/app.py --server.port 8502
# ... work ...
# Ctrl+C
docker-compose build
docker-compose up -d
```

### ❌ Mistake 2: Forgetting to Rebuild After Code Changes

```bash
# BAD
# Edit ui/app.py locally
docker-compose up -d  # ❌ Using OLD image! No changes visible!
```

```bash
# GOOD
# Edit ui/app.py locally
docker-compose build  # ✅ Rebuild with changes
docker-compose up -d  # ✅ Use NEW image
```

### ❌ Mistake 3: Only Restarting After Dependency Changes

```bash
# BAD
poetry add httpx
docker-compose restart  # ❌ New package NOT installed in container!
```

```bash
# GOOD
poetry add httpx
docker-compose build --no-cache  # ✅ Rebuilds with new dependency
docker-compose up -d
```

### ❌ Mistake 4: Running Both Local and Docker Simultaneously

```bash
# BAD
docker-compose up -d
poetry run streamlit run ui/app.py --server.port 8502
# Port 8501 conflict! Which one am I accessing?
```

```bash
# GOOD
# Choose ONE:

# Option A: Local only
docker-compose down
poetry run streamlit run ui/app.py --server.port 8502

# Option B: Docker only
# Make sure no local dev running
docker-compose up -d
```

### ❌ Mistake 5: Editing Files Inside Container

```bash
# BAD
docker exec -it container bash
vi /app/ui/app.py  # ❌ Changes LOST on restart!
```

```bash
# GOOD
# Edit locally in VSCode/your editor
# ui/app.py
docker-compose build  # ✅ Rebuild with changes
docker-compose up -d
```

---

## Port Management

### Port Reference

| Service | Local Dev | Docker | Purpose |
|---------|-----------|--------|---------|
| Streamlit | 8502 | 8501 | Web UI |
| FastAPI | 8002 | 8001 | API Server |
| PostgreSQL | 5432 | 5432 (internal) | Database |

### Check What's Using Ports

```bash
# Windows
netstat -ano | findstr ":8501 :8001 :5432"

# Find process by PID (from netstat output)
tasklist | findstr "PID_NUMBER"

# Kill process by PID
taskkill /F /PID PID_NUMBER
```

### Kill All Python Processes (Nuclear Option)

```bash
# Use with caution - kills ALL Python processes!
taskkill /F /IM python.exe
```

---

## Troubleshooting

### Problem: "Port already in use"

**Symptoms:**
```
Error: bind: address already in use
```

**Solution:**
```bash
# 1. Stop Docker
docker-compose down

# 2. Check what's using the port
netstat -ano | findstr ":8501"

# 3. Kill the process
taskkill /F /PID <PID>

# 4. Verify port is free
netstat -ano | findstr ":8501"

# 5. Try again
docker-compose up -d
```

### Problem: "Changes not appearing in Docker"

**Symptoms:**
- Edited code locally
- Started Docker
- Changes not visible

**Solution:**
```bash
# You forgot to rebuild!
docker-compose down
docker-compose build  # ← This is required!
docker-compose up -d
```

### Problem: "Worker container unhealthy"

**Symptoms:**
```
docker ps
# Shows: (unhealthy)
```

**Solution:**
```bash
# Check logs
docker-compose logs worker

# Common causes:
# 1. Missing MPLCONFIGDIR (already fixed in this project)
# 2. Database connection issues
# 3. Missing dependencies

# Fix: Rebuild
docker-compose down
docker-compose build worker
docker-compose up -d
```

### Problem: "Can't access http://localhost:8501"

**Symptoms:**
- Docker shows healthy
- Browser shows "can't connect"

**Solution:**
```bash
# 1. Verify containers are running
docker ps

# 2. Check if port is actually exposed
netstat -ano | findstr ":8501"

# 3. Try different browser or incognito mode

# 4. Check firewall settings

# 5. Restart Docker Desktop (if using)
```

### Problem: "Database connection refused"

**Symptoms:**
```
FATAL: database "hr_attrition_db" does not exist
```

**Solution:**
```bash
# 1. Database needs initialization
docker-compose down -v  # Remove volumes
docker-compose up -d    # Recreate everything

# 2. Or initialize manually
docker-compose exec db psql -U user -d hr_attrition_db
```

---

## Complete Workflow Examples

### Example 1: Adding a New Feature

```bash
# 1. Start local development
docker-compose down
poetry run streamlit run ui/app.py --server.port 8502

# 2. Edit files in VSCode
# ui/app.py - add new feature
# ... save ...
# Streamlit auto-reloads, see changes instantly

# 3. Test locally
# Access http://localhost:8502
# Verify feature works

# 4. Stop local dev
# Ctrl+C

# 5. Test in Docker
docker-compose build
docker-compose up -d

# 6. Test in Docker
# Access http://localhost:8501
# Verify feature still works

# 7. Check logs
docker-compose logs -f streamlit_app

# 8. If everything works, commit
docker-compose down
git add .
git commit -m "Add new feature X"
git push
```

### Example 2: Adding a New Dependency

```bash
# 1. Add dependency
poetry add httpx

# 2. Test locally first
docker-compose down
poetry install  # Install locally
poetry run streamlit run ui/app.py --server.port 8502

# 3. Use the new library in code
# ui/app.py - import httpx, use it
# Test locally...

# 4. Stop local dev
# Ctrl+C

# 5. Rebuild Docker with new dependency
docker-compose build --no-cache  # Clean build required!
docker-compose up -d

# 6. Verify in Docker
docker-compose logs -f streamlit_app
# Should show no import errors

# 7. Commit
docker-compose down
git add pyproject.toml poetry.lock ui/app.py
git commit -m "Add httpx for API calls"
git push
```

### Example 3: Fixing a Bug

```bash
# 1. Reproduce bug in Docker
docker-compose up -d
# Access http://localhost:8501
# Observe bug behavior

# 2. Switch to local dev for debugging
docker-compose down
poetry run streamlit run ui/app.py --server.port 8502

# 3. Debug locally
# Add print statements
# Add breakpoints (if using debugger)
# Fix the bug

# 4. Test fix locally
# Verify bug is gone

# 5. Test fix in Docker
# Ctrl+C
docker-compose build
docker-compose up -d

# 6. Verify fix in Docker
# Access http://localhost:8501
# Confirm bug is fixed

# 7. Commit
docker-compose down
git add .
git commit -m "Fix bug in prediction logic"
git push
```

---

## Pre-Commit Checklist

Before committing any code:

```bash
✓ Code changes tested locally
✓ docker-compose down executed
✓ docker-compose build completed successfully
✓ docker-compose up -d started all containers
✓ All containers show healthy status
✓ Tested functionality at http://localhost:8501
✓ Checked logs for errors: docker-compose logs
✓ No warnings or errors in logs
✓ Database migrations work (if applicable)
✓ .env file not committed (should be in .gitignore)
```

---

## Environment Variables

### .env (Docker - Current Setup)

```bash
# This is used by Docker
API_BASE_URL=http://172.20.0.4:8001
API_KEY=changeme
DATABASE_URL=postgresql://user:password@db:5432/hr_attrition_db
```

### .env.local (Local Dev - Create This)

```bash
# Create this file for local development
# It's in .gitignore, won't be committed

API_BASE_URL=http://localhost:8002
API_KEY=changeme
DATABASE_URL=postgresql://user:password@localhost:5432/hr_attrition_db
```

**Usage:**
```bash
# Set environment variables from .env.local
# Then start local dev

# Windows CMD:
for /F "tokens=*" %i in (.env.local) do set %i

# Windows PowerShell:
Get-Content .env.local | ForEach-Object { $var = $_.Split('='); [Environment]::SetEnvironmentVariable($var[0], $var[1], 'Process') }

# Then start
poetry run streamlit run ui/app.py --server.port 8502
```

---

## Key Takeaways

### The Golden Rules

1. **Always run `docker-compose down` before switching to local dev**
2. **Always rebuild Docker after code changes: `docker-compose build`**
3. **Never run local dev and Docker simultaneously**
4. **Test in Docker before committing**

### The Complete Workflow

```
Local Dev (Fast) → Docker Test (Safe) → Commit (Confident)
      ↓                    ↓                    ↓
  Quick edits      Full system test      Ready to deploy
```

### When Something Goes Wrong

```bash
# Nuclear option - clean everything and start fresh
docker-compose down -v
taskkill /F /IM python.exe
netstat -ano | findstr ":8501 :8001"
# Kill any remaining processes
docker-compose build --no-cache
docker-compose up -d
```

---

## Additional Resources

- Docker Documentation: https://docs.docker.com/
- Docker Compose: https://docs.docker.com/compose/
- Streamlit Documentation: https://docs.streamlit.io/
- FastAPI Documentation: https://fastapi.tiangolo.com/

---

## Version History

- **v1.0** - Initial guide based on troubleshooting session
- Covers: Local vs Docker, port conflicts, rebuild strategies, helper scripts

---

**Remember:** Docker images are like compiled binaries. Just like you need to recompile code after changes, you need to rebuild Docker images!

**The mistake that caused all the problems:** Not running `docker-compose down` before starting local development, causing port conflicts and accessing the wrong instance.

**The fix:** Always clean up (run `docker-compose down`) before switching modes!
