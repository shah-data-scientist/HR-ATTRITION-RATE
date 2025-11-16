---
title: HR Attrition Prediction Dashboard
emoji: 👥
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: false
license: apache-2.0
---

# HR Attrition Prediction Dashboard 📊

Interactive Streamlit dashboard for predicting employee attrition risk with AI-powered insights.

## 🚀 Features

- 📤 **Upload Employee Data**: Support for CSV and Excel files
- 🔮 **Real-time Predictions**: Instant attrition risk assessment
- 📈 **SHAP Explainability**: Visual explanations for each prediction
- 📥 **Export Results**: Download predictions as Excel
- 🎯 **Risk Categorization**: Low/Medium/High risk classification
- 📊 **Interactive Visualizations**: Charts and graphs using Plotly

## 🎯 How to Use

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

1. Click "Browse files" to upload your CSV/Excel
2. Review the data preview
3. Click "🔮 Predict Attrition" button
4. View results with risk categories and probabilities

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

## 🔗 API Backend

This dashboard connects to: [HR Attrition API](https://huggingface.co/spaces/YOUR-USERNAME/hr-attrition-api)

You can also use the API directly for programmatic access.

## 💡 Example Data

Not sure what format to use? The app provides sample data you can download and modify.

## 🛠️ Technical Stack

- Streamlit 1.41.1
- Pandas 2.2.3
- Plotly 5.24.1
- Requests 2.32.3

## 📈 Model Details

- **Algorithm**: XGBoost Classifier
- **Features**: 33 engineered features
- **Explainability**: SHAP (SHapley Additive exPlanations)

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

## 🎓 Related Projects

- [API Documentation](https://YOUR-USERNAME-hr-attrition-api.hf.space/docs)
- [GitHub Repository](https://github.com/shah-data-scientist/HR-ATTRITION-RATE)
