import shap
import numpy as np

class SHAP_explainer:
    def __init__(self, features, model, anomaly):
        self.X = features
        self.model = model
        self.iso = anomaly
    
    def explain_anomaly(self):
        anomaly_explainer = shap.TreeExplainer(self.iso)
        self.anomaly_score = anomaly_explainer.shap_values(self.X)
        return self.anomaly_score
    
    def explain_risks(self, targets):
        risk_explainer = shap.TreeExplainer(self.model)
        risk_score_all = risk_explainer.shap_values(self.X)
        if len(np.unique(targets)) == 2:
            self.risk_score = risk_score_all
        else:
            self.risk_score = np.array(
                [risk_score_all[i, :, targets[i]] for i in range(len(targets))]
            )
        return self.risk_score

    def generate_shap_paragraph(self, shap_values, feature_names, top_k):
        FEATURE_MAP = {
            'Container_ID': 'Container ID',
            'Declaration_Date (YYYY-MM-DD)': 'Declaration Date',
            'Declaration_Time': 'Declaration Time',
            'Trade_Regime (Import / Export / Transit)': 'Trade Regime',
            'Origin_Country': 'Origin Country',
            'Destination_Port': 'Destination Port',
            'Destination_Country': 'Destination Country',
            'HS_Code': 'HS Code',
            'Importer_ID': 'Importer ID',
            'Exporter_ID': 'Exporter ID',
            'Declared_Value': 'Declared Value',
            'Declared_Weight': 'Declared Weight',
            'Measured_Weight': 'Measured Weight',
            'Shipping_Line': 'Shipping Line',
            'Dwell_Time_Hours': 'Dwell Time (hours)'
        }

        def categorize_impact(value):
            if value < 0.2:
                return "very low"
            elif value < 0.4:
                return "low"
            elif value < 0.6:
                return "moderate"
            elif value < 0.8:
                return "high"
            else:
                return "very high"

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

        # Build sentences
        sentences = []
        for idx in top_indices:
            direction = "increases" if shap_values[:, idx].mean() > 0 else "decreases"
            magnitude = mean_abs_shap[idx] / mean_abs_shap.max()  # normalize 0–1
            impact = categorize_impact(magnitude)
            feature_name = FEATURE_MAP.get(feature_names[idx], feature_names[idx])
            sentences.append(f"{feature_name} {direction} the prediction with {impact} impact.")

        # Combine into paragraph
        paragraph = "The top contributing features are: " + "; ".join(sentences)
        return paragraph
    
    def SHAP_summary(self, feature_names, top_k_risk = 3, top_risk_anomaly = 3):
        self.SHAP_summary_risk = np.array([self.generate_shap_paragraph(self.risk_score[i].reshape(1, -1), feature_names=feature_names, top_k=top_k_risk)
            for i in range(self.risk_score.shape[0])
        ])
        self.SHAP_summary_anomaly = np.array([self.generate_shap_paragraph(self.anomaly_score[i].reshape(1, -1), feature_names=feature_names, top_k=top_risk_anomaly)
            for i in range(self.anomaly_score.shape[0])
        ])
        return self.SHAP_summary_risk, self.SHAP_summary_anomaly
    
    