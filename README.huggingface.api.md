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

# HR Attrition Prediction Platform 🎯

Complete HR employee attrition prediction platform with interactive dashboard and API, all in one Space!

## 🚀 Features

### Interactive Dashboard (Streamlit)
- 📤 **Upload Employee Data**: Support for CSV and Excel files
- 🔮 **Real-time Predictions**: Instant attrition risk assessment
- 📈 **SHAP Explainability**: Visual explanations for each prediction
- 📥 **Export Results**: Download predictions as Excel
- 🎯 **Risk Categorization**: Low/Medium/High risk classification
- 📊 **Interactive Visualizations**: Charts and graphs using Plotly

### API Backend (FastAPI)
- **Real-time Predictions**: Predict employee attrition risk
- **Batch Processing**: Handle multiple employees at once
- **SHAP Explainability**: Generate visual explanations for predictions
- **Excel Export**: Download predictions in Excel format
- **RESTful API**: Well-documented endpoints

## 🌐 Access Points

- **Dashboard (Main)**: https://YOUR-USERNAME-hr-attrition.hf.space (port 7860)
- **API Docs**: https://YOUR-USERNAME-hr-attrition.hf.space:8001/docs (if accessible)
- **API Health**: Use the dashboard or call API at localhost:8001 internally

## 📋 How to Use the Dashboard

### Step 1: Prepare Your Data

Your data file (CSV or Excel) should contain these columns:

**Employee Information:**
- `id_employee` - Unique employee ID
- `age` - Employee age
- `genre` - Gender (M/F or Homme/Femme)
- `revenu_mensuel` - Monthly income
- `statut_marital` - Marital status (Célibataire/Marié(e)/Divorcé(e))
- `departement` - Department (Commercial, Consulting, etc.)
- `poste` - Job position

**Work Details:**
- `annees_dans_l_entreprise` - Years at company
- `annees_dans_le_poste_actuel` - Years in current role
- `heures_supplementaires` - Overtime (Oui/Non)
- `niveau_hierarchique_poste` - Job level (1-5)
- `nombre_experiences_precedentes` - Number of previous jobs
- `annee_experience_totale` - Total work experience
- `distance_domicile_travail` - Distance from home (km)

**Performance & Satisfaction:**
- `satisfaction_employee_environnement` - Environment satisfaction (1-4)
- `satisfaction_employee_nature_travail` - Job satisfaction (1-4)
- `satisfaction_employee_equipe` - Team satisfaction (1-4)
- `satisfaction_employee_equilibre_pro_perso` - Work-life balance (1-4)
- `note_evaluation_precedente` - Previous evaluation score (1-4)
- `note_evaluation_actuelle` - Current evaluation score (1-4)
- `augementaton_salaire_precedente` - Last salary increase (e.g., "15 %")

**Development:**
- `nb_formations_suivies` - Training courses attended
- `nombre_participation_pee` - Participation in profit sharing
- `annees_depuis_la_derniere_promotion` - Years since promotion

**Other:**
- `niveau_education` - Education level (1-5)
- `domaine_etude` - Field of study
- `ayant_enfants` - Has children (Y/N)
- `frequence_deplacement` - Travel frequency (Aucun/Occasionnel/Frequent)
- `nombre_employee_sous_responsabilite` - Number of direct reports
- `annes_sous_responsable_actuel` - Years under current manager

### Step 2: Upload & Predict

1. Visit the Space URL
2. Upload your CSV/Excel file
3. Review the data preview
4. Click "🔮 Predict Attrition" button
5. View results with risk categories and probabilities

### Step 3: Analyze Results

- **Risk Distribution**: See how many employees are in each risk category
- **SHAP Explanations**: Click to generate visual explanations
- **Download Results**: Export predictions to Excel

## 📊 Risk Categories

| Category | Probability | Meaning |
|----------|-------------|---------|
| 🟢 Low | < 30% | Low risk of leaving |
| 🟡 Medium | 30% - 60% | Moderate risk |
| 🔴 High | ≥ 60% | High risk of attrition |

## 🔧 API Endpoints

### Health Check
```
GET http://localhost:8001/health
```

### Predict Attrition
```
POST http://localhost:8001/predict/attrition
```

### Download Excel
```
POST http://localhost:8001/predict/attrition/excel
```

### SHAP Explanation
```
POST http://localhost:8001/predict/shap
```

## 💻 API Usage Example

```python
import requests

url = "http://localhost:8001/predict/attrition"

data = {
    "employees": [
        {
            "id_employee": "EMP001",
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 5000,
            # ... add all required fields
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

## 📈 Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 33 engineered features
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Test Coverage**: 76%

## 🛠️ Technical Stack

- **Frontend**: Streamlit 1.41.1
- **Backend**: FastAPI 0.115.6
- **ML**: XGBoost 2.1.3, SHAP 0.46.0
- **Data**: Pandas 2.2.3
- **Database**: SQLite (local)
- **Deployment**: Docker with Poetry

## 🏗️ Architecture

This Space runs both the API and UI in a single container:
- **Streamlit Dashboard** on port 7860 (main interface)
- **FastAPI Backend** on port 8001 (internal)
- **SQLite Database** for storing predictions
- **Supervisor** manages both processes

## 🔒 Privacy

- No data is stored permanently
- All processing happens in-memory
- Predictions are not logged

## 📝 License

Apache 2.0

## 👥 Authors

shah-data-scientist

## 🐛 Issues & Support

For issues or questions, please visit the [GitHub repository](https://github.com/shah-data-scientist/HR-ATTRITION-RATE)


FastAPI backend for HR employee attrition prediction with SHAP explainability.

## 🚀 Features

- **Real-time Predictions**: Predict employee attrition risk
- **Batch Processing**: Handle multiple employees at once
- **SHAP Explainability**: Generate visual explanations for predictions
- **Excel Export**: Download predictions in Excel format
- **RESTful API**: Well-documented endpoints

## 📋 API Endpoints

### Health Check
```
GET /health
```
Returns API status and model information.

### Root
```
GET /
```
API welcome message.

### Predict Attrition
```
POST /predict/attrition
```
Predict attrition for one or more employees.

**Request Body:**
```json
{
  "employees": [
    {
      "id_employee": "EMP001",
      "age": 35,
      "genre": "M",
      "revenu_mensuel": 5000,
      "satisfaction_employee_environnement": 3,
      "note_evaluation_precedente": 3,
      "departement": "Commercial",
      "poste": "Cadre Commercial",
      "statut_marital": "Marié(e)",
      "niveau_hierarchique_poste": 2,
      "annees_dans_l_entreprise": 5,
      "heures_supplementaires": "Non",
      "nombre_experiences_precedentes": 2,
      "annee_experience_totale": 10,
      "annees_dans_le_poste_actuel": 3,
      "nombre_participation_pee": 1,
      "nb_formations_suivies": 3,
      "nombre_employee_sous_responsabilite": 5,
      "distance_domicile_travail": 10,
      "niveau_education": 3,
      "domaine_etude": "Sciences",
      "ayant_enfants": "Y",
      "frequence_deplacement": "Occasionnel",
      "annees_depuis_la_derniere_promotion": 1,
      "annes_sous_responsable_actuel": 2,
      "augementaton_salaire_precedente": "15 %",
      "satisfaction_employee_nature_travail": 4,
      "satisfaction_employee_equipe": 3,
      "satisfaction_employee_equilibre_pro_perso": 3,
      "note_evaluation_actuelle": 3
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id_employee": "EMP001",
      "attrition_probability": 0.23,
      "risk_category": "Low",
      "model_version": "1.0"
    }
  ],
  "metadata": {
    "total_predictions": 1,
    "timestamp": "2025-11-17T12:00:00"
  }
}
```

### Download Excel
```
POST /predict/attrition/excel
```
Same as `/predict/attrition` but returns Excel file for download.

### SHAP Explanation
```
POST /predict/shap
```
Generate SHAP explanation images for predictions.

## 🔧 Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: 33 engineered features
- **Test Coverage**: 76%
- **Training Data**: HR employee records

### Risk Categories
- **Low Risk**: Probability < 30%
- **Medium Risk**: 30% ≤ Probability < 60%
- **High Risk**: Probability ≥ 60%

## 💻 Usage Examples

### Python
```python
import requests

url = "https://YOUR-USERNAME-hr-attrition-api.hf.space/predict/attrition"

data = {
    "employees": [
        {
            "id_employee": "EMP001",
            "age": 35,
            "genre": "M",
            "revenu_mensuel": 5000,
            # ... add all required fields
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "https://YOUR-USERNAME-hr-attrition-api.hf.space/predict/attrition" \
  -H "Content-Type: application/json" \
  -d '{
    "employees": [{
      "id_employee": "EMP001",
      "age": 35,
      "genre": "M",
      "revenu_mensuel": 5000
    }]
  }'
```

### JavaScript
```javascript
const response = await fetch(
  'https://YOUR-USERNAME-hr-attrition-api.hf.space/predict/attrition',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      employees: [{
        id_employee: 'EMP001',
        age: 35,
        genre: 'M',
        revenu_mensuel: 5000
        // ... other fields
      }]
    })
  }
);
const data = await response.json();
console.log(data);
```

## 📊 Interactive UI

For a user-friendly interface, check out the companion Streamlit app:
👉 [HR Attrition Dashboard](https://huggingface.co/spaces/YOUR-USERNAME/hr-attrition-ui)

## 🛠️ Technical Stack

- FastAPI 0.115.6
- XGBoost 2.1.3
- SHAP 0.46.0
- Pandas 2.2.3
- SQLAlchemy 2.0.36

## 📝 License

Apache 2.0

## 👥 Authors

shah-data-scientist

## 🐛 Issues & Support

For issues or questions, please visit the [GitHub repository](https://github.com/shah-data-scientist/HR-ATTRITION-RATE)
