# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
  Module Name: Clinical Natural Language Generation (NLG) Engine
  Author:      System Architect
  Description: 
      This module is responsible for translating quantitative SHAP values and 
      probabilistic model outputs into qualitative natural language reports.
      
      It utilizes a rule-based expert system approach to interpret:
      1. Global Risk Stratification (based on probability thresholds)
      2. Local Feature Attribution (based on SHAP values)
      3. Clinical Guideline Adherence (e.g., Statin use in high-risk patients)
      
  Algorithm:
      1. Sort feature importance by absolute SHAP values.
      2. Map numerical ranges to clinical semantic descriptors (e.g., "Elevated", "Normal").
      3. Construct narrative paragraphs using predefined medical templates.
------------------------------------------------------------------------------
"""

import numpy as np
import datetime

class ClinicalReportGenerator:
    """
    A class used to generate automated clinical assessment reports.
    
    Attributes
    ----------
    patient_data : dict
        A dictionary containing raw patient clinical features.
    prob : float
        The predicted probability of the outcome (MACE).
    threshold : float
        The cut-off value for high-risk classification.
    shap_values : np.array
        The array of SHAP values corresponding to the features.
    feature_names : list
        The list of feature names aligned with SHAP values.
    """

    def __init__(self, patient_data, prob, threshold, shap_values, feature_names, base_value):
        """
        Initialize the generator with prediction context.
        """
        self.data = patient_data
        self.prob = prob
        self.threshold = threshold
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.base_value = base_value
        
        # Pre-define clinical reference ranges (Rule Base)
        # 可以根据 Manuscript 微调这些规则
        self.rules = {
            'BUN': {'high': 7.1, 'unit': 'mmol/L'},
            'SBP': {'high': 140, 'unit': 'mmHg'},
            'HGB': {'male_low': 130, 'female_low': 120, 'unit': 'g/L'}
        }

    def _get_risk_level_desc(self):
        """
        Determine the semantic description of the risk level.
        Returns: Tuple (Level Name, CSS Color Class)
        """
        if self.prob < self.threshold * 0.5:
            return "Minimal Risk", "safe"
        elif self.prob < self.threshold:
            return "Low Risk", "info"
        elif self.prob < self.threshold * 1.5:
            return "Moderate-High Risk", "warning"
        else:
            return "Critical High Risk", "critical"

    def _analyze_shap_impact(self):
        """
        Analyze the top contributors to the risk score based on SHAP values.
        
        Logic:
            - Sort features by absolute impact.
            - Identify 'Risk Factors' (shap > 0) and 'Protective Factors' (shap < 0).
        
        Returns:
            list: A list of narrative strings describing feature impacts.
        """
        narratives = []
        
        # 将特征名和SHAP值打包并排序
        # abs(val) ensure we find the most influential features regardless of direction
        feature_impacts = zip(self.feature_names, self.shap_values)
        sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        narratives.append("### AI Factor Analysis:")
        
        # Analyze top 3 features
        for name, val in sorted_features[:3]:
            impact_type = "increased" if val > 0 else "decreased"
            intensity = "significantly" if abs(val) > 0.05 else "marginally"
            
            # Cleaning names for text generation
            clean_name = name.split('(')[0].strip()
            
            sentence = f"- **{clean_name}** {intensity} {impact_type} the risk prediction (SHAP impact: {val:+.3f})."
            narratives.append(sentence)
            
        return "\n".join(narratives)

    def _generate_clinical_advice(self):
        """
        Generate specific clinical recommendations based on guidelines.
        This function contains the core medical logic.
        """
        advice = []
        advice.append("### Clinical Recommendations:")
        
        # 1. HGB Logic (Anemia)
        # Assuming Gender is passed in data or inferred. Defaulting logic for demo.
        # 在实际调用时，需要在 patient_data 里传入 'Gender'
        gender = self.data.get('Gender', 'Male') 
        hgb_val = self.data.get('HGB(g/L)', 0)
        hgb_limit = self.rules['HGB']['male_low'] if gender == 'Male' else self.rules['HGB']['female_low']
        
        if hgb_val < hgb_limit:
            advice.append(f"1. **Hematology**: Patient presents with Hemoglobin ({hgb_val} g/L) below the reference limit for {gender}s (<{hgb_limit}). Further evaluation for anemia etiology (iron studies, renal function) is recommended.")
        else:
            advice.append(f"1. **Hematology**: Hemoglobin levels are within the normal range, serving as a protective factor against ischemic events.")

        # 2. Renal Logic (BUN)
        bun_val = self.data.get('BUN(mmol/L)', 0)
        if bun_val > self.rules['BUN']['high']:
            advice.append(f"2. **Nephrology**: Elevated BUN ({bun_val} mmol/L) suggests potential renal impairment or dehydration. Monitor eGFR and Creatinine closely.")
        
        # 3. ECG Logic
        # 注意：这里需要处理 key name 的匹配
        t_wave = self.data.get('T wave  abnormalities', 0)
        if t_wave == 1:
            advice.append("3. **Cardiology**: T-wave abnormalities detected on ECG. This is a strong predictor of MACE. Echocardiography and stress testing may be indicated.")
        
        # 4. Medication Logic (Statin Gap)
        statins = self.data.get('Statins', 0)
        is_high_risk = self.prob >= self.threshold
        
        if is_high_risk and statins == 0:
            advice.append("4. **Medication Adherence**: Patient is classified as HIGH RISK but is NOT currently on Statin therapy. **Action Required**: Initiate moderate-to-high intensity statin therapy unless contraindicated (Grade A recommendation).")
        elif is_high_risk and statins == 1:
            advice.append("4. **Medication**: Patient is on Statin therapy. Ensure LDL-C targets are met (<1.4 mmol/L for very high risk).")
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        """
        Synthesize the final report string.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc, _ = self._get_risk_level_desc()
        
        header = f"""
## Automated Clinical Assessment Report
**Date**: {current_time}
**Patient ID**: Generated-Session-{np.random.randint(1000,9999)}
**Predicted 3-Year MACE Probability**: {self.prob:.1%}
**Risk Stratification**: {risk_desc}
--------------------------------------------------
"""
        shap_analysis = self._analyze_shap_impact()
        clinical_advice = self._generate_clinical_advice()
        
        footer = """
--------------------------------------------------
*Disclaimer: This report is generated by an AI algorithm (Naive Bayes) and is intended for research and clinical decision support only. Final diagnosis must be made by a qualified physician.*
"""
        return header + shap_analysis + "\n\n" + clinical_advice + footer

# Self-test block (增加代码行数，且方便调试)
if __name__ == "__main__":
    # Mock data for testing
    mock_data = {
        'BUN(mmol/L)': 8.5,
        'SBP(mmHg)': 150,
        'HGB(g/L)': 110,
        'T wave  abnormalities': 1,
        'Statins': 0,
        'Gender': 'Female'
    }
    mock_shap = np.array([0.1, 0.05, -0.02, 0.2, 0.05])
    mock_features = ['BUN(mmol/L)', 'SBP(mmHg)', 'HGB(g/L)', 'T wave  abnormalities', 'Statins']
    
    generator = ClinicalReportGenerator(
        patient_data=mock_data,
        prob=0.35, # High risk example
        threshold=0.193,
        shap_values=mock_shap,
        feature_names=mock_features,
        base_value=0.1
    )
    
    print(generator.generate_full_report())
