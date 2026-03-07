# SmartContainer

SmartContainer is a Python library for feature engineering, ML model operations, and explainability on Cargo Container datasets (HackAMineD 2026, Track - InTech, Nirma University).

It provides modular utilities for:
- Feature engineering
- ML modeling (risk classification & anomaly detection)
- Human-readable SHAP summaries

---

## Modules

### 1. Feature_Engineer
Handles feature engineering operations.

```python
class FeatureEngineer(file_path):
```

**Attributes:**
- `self.file_path` – Path to the CSV dataset.
- `self.df` – Pandas dataframe of the input dataset.

**Methods:**
- `engineer_features()`  
  - Performs feature engineering.  
  - Outputs:
    - `X` – Input features for ML model predictions  
    - `y1` – Target categories (Clear / Not Clear)  
    - `y2` – Target categories (No Risk / Low Risk / High Risk)  

---

### 2. Models
Holds ML utilities, including XGBoost for classification and IsolationForest for anomaly detection.

**Attributes:**
- `self.model` – Prediction model (defaults to `XGBClassifier` if not provided).  
- `self.isolator` – IsolationForest model for anomaly detection.  

**Methods:**
- `train_model(X, y, test_size=0.2, stratify=True)`  
  Trains the model with optional stratified K-Fold cross-validation.  
- `predict(X)`  
  Returns predictions, probabilities, and weighted risk score.  
- `anomaly_detection(X)`  
  Returns anomaly scores using IsolationForest.  
- `new_model()`  
  Instantiate a new XGBoostClassifier model.

---

### 3. Summarizer
Provides explainability utilities using SHAP.

**Attributes:**
- `self.X` – Input features used for training/prediction.
- `self.model` – Model used for predictions.
- `self.iso` – IsolationForest used for anomaly detection.

**Methods:**
- `explain_anomaly()`  
  Computes SHAP contributions of features for anomaly scores.  
- `explain_risk(targets)`  
  Computes SHAP contributions of features for risk scores.  
- `generate_paragraph(shap_values, feature_names, top_k)`  
  Generates human-readable, rule-based insights for each record.  
- `SHAP_summary(feature_names, top_k_risk=3, top_k_anomaly=3)`  
  Produces arrays of SHAP summaries for both risk and anomaly analyses.

---

## Usage Example

```python
from smartcontainer import FeatureEngineer, Models, Summarizer

# Load and preprocess data
fe = FeatureEngineer("data.csv")
X, y1, y2 = fe.engineer_features()

# Train risk model
model = Models()
model.train_model(X, y2)

# Predict and get scores
y_pred, probs, risk_score = model.predict(X)
anomaly_score = model.anomaly_detection(X)

# Generate SHAP explanations
summarizer = Summarizer(X, model.model, model.isolator)
risk_shap, anomaly_shap = summarizer.SHAP_summary(feature_names=X.columns.tolist())
```

---

## Notes
- Input datasets must be in **CSV format**.
- The library supports **custom models** and **batch processing**.
- SHAP explanations are **per-record**