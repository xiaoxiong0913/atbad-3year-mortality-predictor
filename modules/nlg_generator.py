# -*- coding: utf-8 -*-
import datetime
import numpy as np

class ClinicalReportGenerator:
    """
    ATBAD Specific Clinical Report Generator (SVM Model)
    """

    def __init__(self, patient_data, prob, threshold, shap_values, feature_names, base_value):
        self.data = patient_data
        self.prob = prob
        self.threshold = threshold
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.base_value = base_value
        
        # ATBAD 临床参考阈值
        self.rules = {
            'BUN': {'high': 8.0, 'unit': 'mmol/L'},
            'HGB': {'low': 120, 'unit': 'g/L'},
            'HR': {'tachy': 100, 'brady': 60, 'unit': 'bpm'},
            'Hosp': {'long': 14, 'unit': 'days'}
        }

    def _get_risk_level_desc(self):
        if self.prob < self.threshold * 0.5: return "Very Low Risk"
        elif self.prob < self.threshold: return "Low Risk"
        elif self.prob < self.threshold * 1.5: return "Moderate Risk"
        else: return "High Risk"

    def _format_patient_info(self):
        age = self.data.get('age', 0)
        hr = self.data.get('HR', 0)
        bun = self.data.get('BUN', 0)
        hgb = self.data.get('HGB', 0)
        hosp = self.data.get('hospitalization', 0)
        
        chd = "Yes" if self.data.get('coronary heart disease') == 1 else "No"
        renal = "Yes" if self.data.get('renal dysfunction') == 1 else "No"
        
        table = f"""
| Clinical Parameter | Result | Reference |
| :--- | :--- | :--- |
| **Age (years)** | {age} | - |
| **Heart Rate (bpm)** | **{hr}** | 60-100 |
| **BUN (mmol/L)** | **{bun}** | 2.8-7.1 |
| **Hemoglobin (g/L)** | **{hgb}** | >120 |
| **Hospitalization (days)** | {hosp} | - |
| **Coronary Heart Disease** | {chd} | Absent |
| **Renal Dysfunction** | {renal} | Absent |
"""
        return table

    def _analyze_shap_impact(self):
        narratives = []
        
        # 清洗数据
        clean_shap_values = []
        for v in self.shap_values:
            if isinstance(v, (np.ndarray, list)):
                # 尝试获取标量值
                if hasattr(v, 'item'): 
                     clean_shap_values.append(v.item())
                elif len(v) > 0:
                     clean_shap_values.append(float(v[0]))
                else:
                     clean_shap_values.append(0.0)
            else:
                clean_shap_values.append(float(v))
        
        feature_impacts = zip(self.feature_names, clean_shap_values)
        sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        narratives.append("#### AI Risk Factor Analysis")
        narratives.append("The AI model identified the following top contributors to the mortality risk:")
        
        for name, val in sorted_features[:3]:
            impact_type = "increased" if val > 0 else "decreased"
            clean_name = name.split('(')[0].strip()
            narratives.append(f"- **{clean_name}** has {impact_type} the estimated risk (Impact score: `{val:+.3f}`).")
            
        return "\n".join(narratives)

    def _generate_clinical_advice(self):
        advice = []
        advice.append("#### Clinical Management Recommendations")
        
        hr = self.data.get('HR', 0)
        if hr > self.rules['HR']['tachy']:
            advice.append(f"1. **Hemodynamics**: Tachycardia ({hr} bpm) detected. Aggressive rate control (Beta-blockers) is recommended to reduce aortic wall stress.")
        elif hr < self.rules['HR']['brady']:
            advice.append(f"1. **Hemodynamics**: Bradycardia ({hr} bpm) detected. Monitor for perfusion and adjust AV-nodal blocking agents if necessary.")
        else:
            advice.append("1. **Hemodynamics**: Heart rate within target range. Maintain strict BP control (SBP <120 mmHg).")

        chd = self.data.get('coronary heart disease', 0)
        if chd == 1:
            advice.append("2. **Comorbidity (CHD)**: Patient has Coronary Heart Disease. Optimize antiplatelet therapy and statins. Evaluate for revascularization if symptomatic.")
        
        renal = self.data.get('renal dysfunction', 0)
        bun = self.data.get('BUN', 0)
        if renal == 1 or bun > self.rules['BUN']['high']:
            advice.append(f"3. **Renal Protection**: Renal dysfunction indicated (BUN: {bun} mmol/L). Avoid nephrotoxic agents (contrast media). Consult Nephrology.")
        
        hgb = self.data.get('HGB', 0)
        if hgb < self.rules['HGB']['low']:
            advice.append(f"4. **Hematology**: Low HGB ({hgb} g/L). Investigate for blood loss (dissection extension/rupture) or chronic anemia.")
        
        hosp = self.data.get('hospitalization', 0)
        if hosp > self.rules['Hosp']['long']:
            advice.append(f"5. **Recovery**: Prolonged hospitalization ({hosp} days). Assess for nosocomial complications and rehabilitation needs.")

        # === 修改点：高危提示 ===
        # 使用 HTML span 标签控制颜色为红色 (#dc3545)，加粗，但字号保持默认 (不使用标题)
        if self.prob >= self.threshold:
            advice.append("---")
            alert_text = (
                "<span style='color: #dc3545; font-weight: bold; font-size: 16px;'>"
                "⚠️ High Risk Alert: This patient is in the high-risk group for 3-year mortality. "
                "Consider closer surveillance (CTA every 3-6 months) and aggressive risk factor modification."
                "</span>"
            )
            advice.append(alert_text)
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc = self._get_risk_level_desc()
        
        header = f"""
### ATBAD 3-Year Prognostic Report
**Session ID**: `{session_id}`  |  **Date**: {current_time}

---
**Predicted Mortality Risk**: `{self.prob:.1%}`
**Risk Classification**: **{risk_desc}**
"""
        patient_table = self._format_patient_info()
        shap_analysis = self._analyze_shap_impact()
        clinical_advice = self._generate_clinical_advice()
        
        footer = """
---
*Disclaimer: AI-assisted tool for research reference only. Not a substitute for clinical judgment.*
"""
        return header + patient_table + "\n" + shap_analysis + "\n\n" + clinical_advice + footer
