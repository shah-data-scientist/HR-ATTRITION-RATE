# Quickstart — HR Attrition Rate

Get the system running in under 10 minutes.

## Prerequisites

- Docker & Docker Compose
- Git

Everything else runs inside containers.

---

## 1. Clone

```bash
git clone https://github.com/shah-data-scientist/HR-ATTRITION-RATE.git
cd HR-ATTRITION-RATE
```

## 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
API_KEY=replace_with_64_char_hex_string
SECRET_KEY=replace_with_at_least_32_chars
```

Leave the database variables at their defaults for local development.

## 3. Start services

```bash
docker compose --profile local up -d
```

This starts four containers: PostgreSQL, database initialiser, FastAPI API, Streamlit UI.
First run takes ~3 minutes while images build.

## 4. Open the app

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8581 |
| API interactive docs | http://localhost:8081/docs |
| API health check | http://localhost:8081/health |

Log in with the credentials seeded into the database (see `database/seed_data.py`).

## 5. Make a prediction

In the Streamlit UI:
1. Upload the three CSV files from `data/` (`extrait_eval.csv`, `extrait_sirh.csv`, `extrait_sondage.csv`)
2. Click **Predict Attrition**
3. View risk categories, download the Excel report, explore SHAP charts

Or call the API directly:

```bash
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_api_key_here" \
  -d @tests/fixtures/test_payload.json
```

---

## Stop services

```bash
docker compose --profile local down
# Add -v to also remove the database volume
```

---

## Next steps

- [DEVELOPMENT.md](DEVELOPMENT.md) — run locally without Docker, run tests, linting
- [DEPLOYMENT.md](DEPLOYMENT.md) — production deployment
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design overview
