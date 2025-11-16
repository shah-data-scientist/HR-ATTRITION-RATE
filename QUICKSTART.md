# Quick Start Guide - HR Attrition Prediction

Get the HR Attrition Prediction system running in under 5 minutes!

## 🚀 Installation (One-Time Setup)

### Step 1: Install Prerequisites

**Install Poetry:**
```bash
# On Linux/Mac
curl -sSL https://install.python-poetry.org | python3 -

# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

**Install Docker Desktop** (optional, for database):
- Download from https://www.docker.com/products/docker-desktop

### Step 2: Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd hr-attrition-rate

# Install dependencies
poetry install
```

### Step 3: Set Up Database

**Option A: Using Docker (Easiest)**
```bash
docker-compose up db -d
```

**Option B: Using Existing PostgreSQL**
```bash
# Create database
createdb hr_attrition_db
```

### Step 4: Initialize Database

```bash
poetry run python database/init_db.py
```

✅ **Setup Complete!**

---

## 🏃 Running the Application

### For Windows Users:

**Open TWO Command Prompts**

**Command Prompt 1:**
```cmd
cd path\to\hr-attrition-rate
scripts\start-api.bat
```
Wait for "Application startup complete" message

**Command Prompt 2:**
```cmd
cd path\to\hr-attrition-rate
scripts\start-ui.bat
```

### For Linux/Mac Users:

**Open TWO Terminals**

**Terminal 1:**
```bash
cd path/to/hr-attrition-rate
./scripts/start-api.sh
```
Wait for "Application startup complete" message

**Terminal 2:**
```bash
cd path/to/hr-attrition-rate
./scripts/start-ui.sh
```

---

## 🌐 Access the Application

Once both services are running:

- **Open your browser**: http://localhost:8501
- **API Documentation**: http://localhost:8001/docs
- **API Health Check**: http://localhost:8001/health

---

## 📤 Using the Application

### Upload Data

The UI expects 3 CSV files:
1. `extrait_eval.csv` - Employee evaluation data
2. `extrait_sirh.csv` - HR system data  
3. `extrait_sondage.csv` - Employee survey data

**Sample files** are included in the `data/` directory for testing!

### Get Predictions

1. Click "Predict Attrition" button
2. Wait for processing (usually < 10 seconds)
3. Download the Excel report
4. View SHAP explanations for each employee

---

## 🛑 Stopping the Application

Press `Ctrl+C` in each terminal/command prompt where the services are running.

**Stop the database:**
```bash
docker-compose down
```

---

## 🐛 Troubleshooting

### "Connection Refused" Error

**Problem:** UI can't connect to API

**Solution:**
1. Make sure you started the API first (wait for "Application startup complete")
2. Check API is running: Open http://localhost:8001/health in browser
3. Should see: `{"status":"ok","message":"API is healthy"}`

### "Model file not found"

**Problem:** API can't find the trained model

**Solution:** Train the model:
```bash
poetry run python train.py
```

### "Port already in use"

**Problem:** Another application is using port 8001 or 8501

**Solution:**

**Windows:**
```cmd
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -i :8001
kill -9 <PID>
```

### Database Connection Error

**Problem:** Can't connect to PostgreSQL

**Solution:**
1. Check Docker is running: `docker ps`
2. Start database: `docker-compose up db -d`
3. Wait 10 seconds for database to be ready
4. Retry

---

## 🎉 Next Steps

Once you're comfortable with the basics:

- Read [README.md](README.md) for full documentation
- Check [DEVELOPMENT.md](DEVELOPMENT.md) for development workflows
- See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment

---

## 🆘 Still Having Issues?

1. Check the terminal/command prompt for error messages
2. Review the [Troubleshooting](#-troubleshooting) section above
3. Look at archived documentation in `docs/archive/`
4. Open an issue on GitHub with:
   - Your operating system
   - Error messages
   - Steps you've tried

---

## 📋 Quick Command Reference

```bash
# Start everything (after initial setup)
docker-compose up db -d                    # Start database
./scripts/start-api.sh                     # Start API (Terminal 1)
./scripts/start-ui.sh                      # Start UI (Terminal 2)

# Stop everything
# Press Ctrl+C in both terminals
docker-compose down                        # Stop database

# Run tests
poetry run pytest

# Update dependencies
poetry update

# View logs
docker-compose logs -f db                  # Database logs
# API and UI logs are in the terminal windows
```
