# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
  Module Name: Clinical Natural Language Generation (NLG) Engine
  Author:      System Architect
  Description: 
      Translates quantitative SHAP values into qualitative clinical reports.
------------------------------------------------------------------------------
"""

import numpy as np
import datetime

class ClinicalReportGenerator:
    """
    Generates automated clinical assessment reports.
    """

    def __init__(self, patient_data, prob, threshold, shap_values, feature_names, base_value):
        self.data = patient_data
        self.prob = prob
        self.threshold = threshold
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.base_value = base_value
        
        self.rules = {
            'BUN': {'high': 7.1},
            'SBP': {'high': 140},
            'HGB': {'male_low': 130, 'female_low': 120}
        }

    def _get_risk_level_desc(self):
        if self.prob < self.threshold * 0.5: return "Minimal Risk"
        elif self.prob < self.threshold: return "Low Risk"
        elif self.prob < self.threshold * 1.5: return "Moderate-High Risk"
        else: return "Critical High Risk"

    def _analyze_shap_impact(self):
        narratives = []
        feature_impacts = zip(self.feature_names, self.shap_values)
        sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        narratives.append("### AI Factor Analysis:")
        for name, val in sorted_features[:3]:
            impact_type = "increased" if val > 0 else "decreased"
            intensity = "significantly" if abs(val) > 0.05 else "marginally"
            clean_name = name.split('(')[0].strip()
            narratives.append(f"- **{clean_name}** {intensity} {impact_type} the risk prediction (SHAP impact: {val:+.3f}).")
        return "\n".join(narratives)

    def _generate_clinical_advice(self):
        advice = []
        advice.append("### Clinical Recommendations:")
        
        gender = self.data.get('Gender', 'Male') 
        hgb_val = self.data.get('HGB(g/L)', 0)
        hgb_limit = self.rules['HGB']['male_low'] if gender == 'Male' else self.rules['HGB']['female_low']
        
        if hgb_val < hgb_limit:
            advice.append(f"1. **Hematology**: Hemoglobin ({hgb_val} g/L) below limit for {gender}s (<{hgb_limit}). Check for anemia.")
        else:
            advice.append(f"1. **Hematology**: Hemoglobin within normal range.")

        bun_val = self.data.get('BUN(mmol/L)', 0)
        if bun_val > self.rules['BUN']['high']:
            advice.append(f"2. **Nephrology**: Elevated BUN ({bun_val} mmol/L). Monitor renal function.")
        
        t_wave = self.data.get('T wave  abnormalities', 0)
        if t_wave == 1:
            advice.append("3. **Cardiology**: T-wave abnormalities detected. Correlate with symptoms.")
        
        statins = self.data.get('Statins', 0)
        is_high_risk = self.prob >= self.threshold
        
        if is_high_risk and statins == 0:
            advice.append("4. **Medication**: High Risk patient NOT on Statins. Consider initiating therapy.")
        elif is_high_risk and statins == 1:
            advice.append("4. **Medication**: On Statin therapy. Ensure LDL-C targets are met.")
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        """
        Synthesize the final report string with updated footer.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc = self._get_risk_level_desc()
        
        header = f"""
## Automated Clinical Assessment Report
**Date**: {current_time}
**Patient ID**: Generated-Session-{np.random.randint(1000,9999)}
**Predicted Probability**: {self.prob:.1%} ({risk_desc})
--------------------------------------------------
"""
        shap_analysis = self._analyze_shap_impact()
        clinical_advice = self._generate_clinical_advice()
        
        # --- 关键修改处 ---
        footer = """
--------------------------------------------------
*Disclaimer: For Reference Only. Not for primary diagnosis. (C) 2026 Yichang Central People's Hospital.*
"""
        # ----------------
        return header + shap_analysis + "\n\n" + clinical_advice + footer
