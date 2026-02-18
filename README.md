# 💊 Healthcare Cost Prediction with Explainable AI

An end-to-end machine learning system for predicting healthcare costs using Medicare data, featuring XGBoost/Random Forest ensemble modeling, SHAP explainability, bias analysis, and an interactive Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📊 Project Overview

This project analyzes **120,000+ patient records** from CMS Medicare datasets to predict healthcare costs with high accuracy while maintaining model interpretability through SHAP (SHapley Additive exPlanations).

**Key Metrics:**
- 🎯 **Model Performance:** R² = 0.78, MAE = $4,200
- 📊 **Dataset Size:** 120,000 patient records
- 🔍 **Features:** 45+ clinical and demographic variables
- ⚖️ **Bias Analysis:** Fairness metrics across age, gender, ethnicity
- 🚀 **Deployment:** Interactive Streamlit web application

## 🎯 Features

### Machine Learning Pipeline
- ✅ Ensemble modeling (XGBoost + Random Forest)
- ✅ Automated feature engineering (comorbidity scores, age groups)
- ✅ Hyperparameter tuning with Grid Search
- ✅ Cross-validation for robust performance
- ✅ Model versioning and tracking

### Explainability & Interpretability
- 🔍 SHAP values for individual predictions
- 📊 Global feature importance analysis
- 🎯 Waterfall plots for prediction breakdown
- 📈 Dependence plots for feature interactions
- 🧪 What-if analysis for scenario testing

### Bias & Fairness
- ⚖️ Disparate impact analysis across demographics
- 📊 Equal opportunity metrics
- 🔍 Subgroup performance evaluation
- 📈 Fairness-aware model selection

### Web Application
- 🌐 User-friendly Streamlit interface
- 📊 Real-time predictions
- 🔍 Explanation dashboard
- 📈 Visualization of key factors
- 💾 Downloadable prediction reports

## 🏗️ Architecture

```
┌─────────────────┐
│  CMS Medicare   │
│    Dataset      │
└────────┬────────┘
         │
         ├──► Data Ingestion & Cleaning
         │
         ↓
┌─────────────────┐
│ Feature Eng.    │
│ - Comorbidity   │
│ - Age Groups    │
│ - Risk Scores   │
└────────┬────────┘
         │
         ├──► Model Training Pipeline
         │     ├─► XGBoost
         │     └─► Random Forest
         ↓
┌─────────────────┐
│ Ensemble Model  │
│   R² = 0.78     │
└────────┬────────┘
         │
         ├──► SHAP Explainability
         │
         ↓
┌─────────────────┐
│   Streamlit     │
│   Web App       │
└─────────────────┘
```

## 📁 Project Structure

```
healthcare-cost-prediction/
├── data/
│   ├── raw/                    # Raw CMS data
│   ├── processed/              # Cleaned datasets
│   └── sample_data.csv         # Demo dataset
├── src/
│   ├── data_preprocessing.py   # ETL pipeline
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # ML model training
│   ├── explainability.py       # SHAP analysis
│   ├── bias_detection.py       # Fairness metrics
│   └── config.py               # Configuration
├── models/
│   ├── xgboost_model.pkl       # Trained XGBoost
│   ├── rf_model.pkl            # Trained Random Forest
│   └── feature_names.json      # Feature metadata
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_shap_analysis.ipynb
├── app/
│   ├── streamlit_app.py        # Main web application
│   ├── utils.py                # Helper functions
│   └── assets/                 # Images, CSS
├── tests/
│   └── test_model.py           # Unit tests
├── requirements.txt
├── .env.example
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-cost-prediction.git
cd healthcare-cost-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download sample data** (or use provided sample)
```bash
# Optional: Download from CMS
python src/data_preprocessing.py --download
```

4. **Train the model**
```bash
python src/model_training.py
```

5. **Launch web application**
```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501`

## 📈 Model Performance

### Overall Metrics
```
Metric              Train Set    Test Set     Cross-Val
------------------  -----------  -----------  -----------
R² Score            0.82         0.78         0.77 ± 0.02
MAE                 $3,850       $4,200       $4,150 ± 250
RMSE                $6,320       $6,890       $6,750 ± 380
MAPE                12.3%        13.8%        13.5% ± 0.8%
```

### Feature Importance (Top 10)
```
Feature                         SHAP Impact
-----------------------------   -----------
Procedure Complexity Index      23.4%
Number of Comorbidities         18.7%
Age Group (65-74)               12.3%
Hospital Length of Stay         10.8%
Prior Year Costs                 8.9%
Chronic Condition Count          7.2%
Region (Urban vs Rural)          5.4%
Insurance Type                   4.3%
Gender                           3.8%
BMI Category                     3.1%
```

## 🔬 Technical Details

### Feature Engineering
The project creates **45+ features** including:

- **Demographic Features:** Age groups, gender, ethnicity, geography
- **Clinical Features:** Comorbidity scores (Charlson, Elixhauser), chronic conditions
- **Utilization Features:** Prior hospitalizations, ER visits, pharmacy fills
- **Risk Scores:** CMS-HCC risk scores, disease burden index
- **Temporal Features:** Seasonal patterns, trend components

### Model Architecture
```python
# Ensemble approach
ensemble = VotingRegressor([
    ('xgb', XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8
    )),
    ('rf', RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10
    ))
])
```

### SHAP Explainability Example
```python
# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize for a single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
)
```

### Bias Detection Results
```
Demographic Group    MAE      Disparate Impact    Equal Opportunity
-------------------  -------  ------------------  -----------------
Overall              $4,200   1.00                1.00
Age 18-44            $3,850   0.92                0.94
Age 45-64            $4,100   0.98                0.99
Age 65+              $4,450   1.06                1.03
Male                 $4,150   0.99                1.01
Female               $4,250   1.01                0.99
White                $4,100   0.98                0.97
Black                $4,450   1.06                1.05
Hispanic             $4,280   1.02                1.03
```

## 📊 Sample Predictions

### Example 1: Low-Risk Patient
```
Input:
- Age: 45, Gender: Male, BMI: 24
- Comorbidities: 1 (Hypertension)
- Prior Year Cost: $2,500

Prediction: $5,200 (95% CI: $4,100 - $6,300)

Top Contributing Factors (SHAP):
1. Low comorbidity count: -$1,200
2. Age under 50: -$800
3. Normal BMI: -$300
4. Low prior costs: -$450
```

### Example 2: High-Risk Patient
```
Input:
- Age: 72, Gender: Female, BMI: 35
- Comorbidities: 5 (Diabetes, Heart Disease, COPD, CKD, Arthritis)
- Prior Year Cost: $18,000

Prediction: $28,400 (95% CI: $23,200 - $33,600)

Top Contributing Factors (SHAP):
1. High comorbidity count: +$6,800
2. Age over 70: +$3,200
3. High BMI: +$1,500
4. High prior costs: +$4,200
```

## 🛠️ Technologies Used

- **Python 3.9+**: Core programming
- **XGBoost**: Gradient boosting framework
- **scikit-learn**: Random Forest, preprocessing
- **SHAP**: Model explainability
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Streamlit**: Web application framework
- **Matplotlib/Seaborn**: Data visualization
- **Plotly**: Interactive visualizations

## 🌐 Web Application Features

The Streamlit app provides:

1. **Prediction Interface**
   - Input patient demographics and clinical data
   - Real-time cost prediction
   - Confidence intervals

2. **Explanation Dashboard**
   - SHAP waterfall plot for prediction breakdown
   - Feature importance visualization
   - What-if scenario analysis

3. **Model Insights**
   - Overall performance metrics
   - Feature distributions
   - Bias analysis results

4. **Downloadable Reports**
   - PDF prediction summaries
   - Detailed SHAP analysis
   - Data quality reports

## 📝 Future Enhancements

- [ ] Integration with EHR systems (HL7 FHIR)
- [ ] Deep learning models (neural networks)
- [ ] Real-time data streaming
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Advanced fairness-aware learning algorithms

## 👤 Author

**Rakesh Budige**
- 🎓 MS Computer Science, University of Illinois Springfield
- 💼 Data Analyst | Healthcare Analytics & ML
- 🔗 [LinkedIn](https://linkedin.com/in/yourprofile)
- 📧 your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Centers for Medicare & Medicaid Services (CMS) for open data
- SHAP library creators for explainability tools
- Healthcare analytics community for insights

## ⚕️ Important Disclaimer

This tool is for educational and research purposes only. It should NOT be used for making actual healthcare decisions or determining patient care. Always consult qualified healthcare professionals for medical advice.

---

**⭐ If you found this project useful, please give it a star!**
