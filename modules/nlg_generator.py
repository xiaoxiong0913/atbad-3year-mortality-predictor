# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
  Module Name: Clinical Natural Language Generation (NLG) Engine
  Author:      System Architect
  Description: 
      Translates quantitative SHAP values into qualitative clinical reports.
      (Professional English Version)
------------------------------------------------------------------------------
"""

import numpy as np
import datetime

class ClinicalReportGenerator:
    """
    Generates automated clinical assessment reports (Professional Edition).
    """

    def __init__(self, patient_data, prob, threshold, shap_values, feature_names, base_value):
        self.data = patient_data
        self.prob = prob
        self.threshold = threshold
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.base_value = base_value
        
        # Clinical Logic Rules (English)
        self.rules = {
            'BUN': {'high': 7.1, 'unit': 'mmol/L'},
            'SBP': {'high': 140, 'unit': 'mmHg'},
            'HGB': {'male_low': 130, 'female_low': 120, 'unit': 'g/L'}
        }

    def _get_risk_level_desc(self):
        """Returns English risk description."""
        if self.prob < self.threshold * 0.5: return "Minimal Risk"
        elif self.prob < self.threshold: return "Low Risk"
        elif self.prob < self.threshold * 1.5: return "Moderate-High Risk"
        else: return "Critical High Risk"

    def _format_patient_info(self):
        """Formats patient data as a Markdown Table for better UI."""
        gender = self.data.get('Gender', 'N/A')
        bun = self.data.get('BUN(mmol/L)', 0)
        sbp = self.data.get('SBP(mmHg)', 0)
        hgb = self.data.get('HGB(g/L)', 0)
        t_wave = "Yes" if self.data.get('T wave  abnormalities') == 1 else "No"
        statins = "Yes" if self.data.get('Statins') == 1 else "No"
        
        table = f"""
| Parameter | Value | Reference |
| :--- | :--- | :--- |
| **Gender** | {gender} | - |
| **SBP** | {sbp} mmHg | <140 |
| **BUN** | {bun} mmol/L | 2.8-7.1 |
| **HGB** | {hgb} g/L | M>130, F>120 |
| **T-Wave** | {t_wave} | Absent |
| **Statins** | {statins} | - |
"""
        return table

    def _analyze_shap_impact(self):
        """Generates bullet points for SHAP analysis."""
        narratives = []
        feature_impacts = zip(self.feature_names, self.shap_values)
        sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        narratives.append("#### 🧠 AI Factor Analysis")
        for name, val in sorted_features[:3]:
            impact_type = "increased ⬆️" if val > 0 else "decreased ⬇️"
            clean_name = name.split('(')[0].strip()
            # 格式化显示：特征名加粗，影响值保留3位小数
            narratives.append(f"- **{clean_name}** has {impact_type} the risk (Impact: `{val:+.3f}`).")
            
        return "\n".join(narratives)

    def _generate_clinical_advice(self):
        """Generates clinical advice based on rules."""
        advice = []
        advice.append("#### 🩺 Clinical Recommendations")
        
        gender = self.data.get('Gender', 'Male') 
        hgb_val = self.data.get('HGB(g/L)', 0)
        hgb_limit = self.rules['HGB']['male_low'] if gender == 'Male' else self.rules['HGB']['female_low']
        
        if hgb_val < hgb_limit:
            advice.append(f"1. **Hematology**: HGB is **{hgb_val} g/L** (Low). Evaluate for anemia.")
        else:
            advice.append(f"1. **Hematology**: HGB levels are within normal range.")

        bun_val = self.data.get('BUN(mmol/L)', 0)
        if bun_val > self.rules['BUN']['high']:
            advice.append(f"2. **Nephrology**: Elevated BUN (**{bun_val} mmol/L**). Check eGFR/Creatinine.")
        
        t_wave = self.data.get('T wave  abnormalities', 0)
        if t_wave == 1:
            advice.append("3. **Cardiology**: **T-wave abnormalities** detected. Correlate with ischemic symptoms.")
        
        statins = self.data.get('Statins', 0)
        is_high_risk = self.prob >= self.threshold
        
        if is_high_risk and statins == 0:
            advice.append("4. **Medication**: Patient is **High Risk** and NOT on Statins. **Action**: Initiate Statin therapy.")
        elif is_high_risk and statins == 1:
            advice.append("4. **Medication**: Patient is on Statin therapy. Monitor lipid profile.")
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        """
        Synthesize the final report string.
        """
        # 使用真实时间生成 Session ID
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc = self._get_risk_level_desc()
        
        # 头部信息
        header = f"""
### 📋 Automated Clinical Assessment Report
**Session ID**: `{session_id}`  |  **Date**: {current_time}

---
**Predicted Probability**: `{self.prob:.1%}`
**Risk Classification**: **{risk_desc}**
"""
        # 组合各部分
        patient_table = self._format_patient_info()
        shap_analysis = self._analyze_shap_impact()
        clinical_advice = self._generate_clinical_advice()
        
        # 底部免责声明
        footer = """
---
*Disclaimer: For Reference Only. Not for primary diagnosis. (C) 2026 Yichang Central People's Hospital.*
"""
        return header + patient_table + "\n" + shap_analysis + "\n\n" + clinical_advice + footer
