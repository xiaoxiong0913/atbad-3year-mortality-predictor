# -*- coding: utf-8 -*-
import datetime

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
        
        # 医学参考范围 (参考旧版代码)
        self.rules = {
            'BUN': {'high': 8.0, 'unit': 'mmol/L'},  # 正常上限约 7.1-8.0
            'HGB': {'low': 120, 'unit': 'g/L'},
            'HR': {'tachy': 100, 'unit': 'bpm'},
            'Hosp': {'long': 14, 'unit': 'days'}
        }

    def _get_risk_level_desc(self):
        # 根据阈值判断风险等级
        if self.prob < self.threshold * 0.5: return "Very Low Risk"
        elif self.prob < self.threshold: return "Low Risk"
        elif self.prob < self.threshold * 1.5: return "Moderate Risk"
        else: return "High Risk"

    def _format_patient_info(self):
        """
        ATBAD Patient Profile Table
        """
        age = self.data.get('age', 0)
        hr = self.data.get('HR', 0)
        bun = self.data.get('BUN', 0)
        hgb = self.data.get('HGB', 0)
        hosp = self.data.get('hospitalization', 0)
        
        # 处理二分类变量 (0/1 -> Yes/No)
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
        # SHAP 分析文本生成
        narratives = []
        feature_impacts = zip(self.feature_names, self.shap_values)
        # 按绝对值排序
        sorted_features = sorted(feature_impacts, key=lambda x: abs(x[1]), reverse=True)
        
        narratives.append("#### 🧠 AI Risk Factor Analysis")
        narratives.append("The AI model identified the following top contributors to the mortality risk:")
        
        for name, val in sorted_features[:3]:
            impact_type = "increased ⬆️" if val > 0 else "decreased ⬇️"
            # 清洗特征名 (去掉可能的单位后缀)
            clean_name = name.split('(')[0].strip()
            narratives.append(f"- **{clean_name}** has {impact_type} the estimated risk (Impact score: `{val:+.3f}`).")
            
        return "\n".join(narratives)

    def _generate_clinical_advice(self):
        """
        基于 ATBAD 指南的个性化建议
        """
        advice = []
        advice.append("#### 🩺 Clinical Management Recommendations")
        
        # 1. 基础管理
        advice.append("1. **General Targets**: Strict BP control (SBP <120 mmHg) and HR control (<60-70 bpm) are fundamental.")

        # 2. 冠心病逻辑
        chd = self.data.get('coronary heart disease', 0)
        if chd == 1:
            advice.append("2. **Comorbidity Management (CHD)**: Patient has Coronary Heart Disease. Optimize antiplatelet therapy and statins. Evaluate for revascularization if symptomatic.")
        
        # 3. 肾功能逻辑
        renal = self.data.get('renal dysfunction', 0)
        bun = self.data.get('BUN', 0)
        if renal == 1 or bun > self.rules['BUN']['high']:
            advice.append(f"3. **Renal Protection**: Renal dysfunction indicated (BUN: {bun} mmol/L). Avoid nephrotoxic agents (contrast media). Consult Nephrology.")
        
        # 4. 贫血/血红蛋白
        hgb = self.data.get('HGB', 0)
        if hgb < self.rules['HGB']['low']:
            advice.append(f"4. **Hematology**: Low HGB ({hgb} g/L). Investigate for blood loss or chronic anemia, which may worsen prognosis.")
        
        # 5. 住院时长逻辑
        hosp = self.data.get('hospitalization', 0)
        if hosp > self.rules['Hosp']['long']:
            advice.append(f"5. **Recovery**: Prolonged hospitalization ({hosp} days). Assess for nosocomial complications and need for comprehensive rehabilitation.")

        # 高危特别提示
        if self.prob >= self.threshold:
            advice.append("---")
            advice.append("⚠️ **High Risk Alert**: This patient is in the high-risk group for 3-year mortality. Consider closer surveillance (CTA every 3-6 months) and aggressive risk factor modification.")
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc = self._get_risk_level_desc()
        
        header = f"""
### 📋 ATBAD 3-Year Prognostic Report
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
