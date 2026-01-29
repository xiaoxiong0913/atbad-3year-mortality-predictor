import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
import datetime

# ================= 1. 引用自定义模块 =================
from modules.database import PatientDatabase
from modules.nlg_generator import ClinicalReportGenerator
from modules.pdf_report import PDFReportEngine
from modules.batch_processor import BatchProcessor
from modules.analytics import AnalyticsEngine

# ================= 2. 系统初始化与配置 =================
st.set_page_config(
    page_title="ATBAD Mortality Risk Prediction Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载外部 CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .protocol-card { padding: 15px; border-radius: 4px; margin-bottom: 15px; background: #f8f9fa; border: 1px solid #dee2e6; }
            .info-card { border-left: 5px solid #004085; background-color: #cce5ff; color: #004085; padding: 10px; }
            .header-text { font-family: 'Times New Roman', serif; }
        </style>
        """, unsafe_allow_html=True)

local_css("assets/style.css")

# ================= 3. 资源加载 =================
@st.cache_resource
def load_system():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    try:
        with open(os.path.join(ASSETS_DIR, "svm_model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        
        imputer = None
        if os.path.exists(os.path.join(ASSETS_DIR, "imputer.pkl")):
            with open(os.path.join(ASSETS_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
            
        return model, scaler, imputer
    except Exception as e:
        st.error(f"System Error: Failed to load core assets. {e}")
        return None, None, None

model, scaler, imputer = load_system()
db = PatientDatabase()

THRESHOLD = 0.5 

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    st.header("Navigation")
    
    page = st.radio(
        "Go to", 
        ["Individual Assessment", "Batch Analysis", "Clinical Dashboard", "Project Introduction"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**System Status**")
    if model:
        st.success("Model Loaded")
        st.info("Database Connected")
    else:
        st.error("System Offline")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 单例预测 (布局调整版) -----------------
if page == "Individual Assessment":
    st.title("Individual Risk Assessment")
    
    st.markdown("<div class='protocol-card'><b>Protocol Note:</b> Please ensure all input values are collected at the time of admission or diagnosis.</div>", unsafe_allow_html=True)

    # === 布局：左侧表单，右侧说明 (2:1 比例) ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("input_form_atbad"):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Demographics & Vitals")
                age = st.number_input("Age (years)", 20, 100, 60)
                hr = st.number_input("Heart Rate (bpm)", 30, 180, 80)
                hosp = st.number_input("Hospitalization (days)", 1, 100, 10)
                
                st.subheader("Comorbidities")
                chd = st.selectbox("Coronary Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                
            with c2:
                st.subheader("Laboratory Markers")
                bun = st.number_input("BUN (mmol/L)", 0.1, 100.0, 7.0, 0.1)
                hgb = st.number_input("Hemoglobin (g/L)", 30, 250, 130)
                
                st.subheader("Renal Status")
                renal = st.selectbox("Renal Dysfunction", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Calculate Risk", type="primary")

    with col2:
        # === 按照您要求的样式放置右侧说明 ===
        st.markdown("<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #ddd;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:10px; margin-top:0;'>3-Year Mortality Prediction</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:10px; color:#555;'>Model Overview</h4>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:14px; line-height: 1.6;'>
        This predictive tool uses an <b>SVM machine learning model</b> to estimate 3-year mortality risk in patients with acute Type B aortic dissection.<br><br>
        <b>Performance Metrics:</b><br>
        - AUC: <b>0.94</b><br>
        - Accuracy: <b>88.8%</b><br>
        - Risk Threshold: <b>0.207</b>
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and model:
        # 特征顺序必须与 scaler.pkl 一致
        cols = ['age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction']
        
        inputs = {
            'age': age,
            'HR': hr,
            'BUN': bun,
            'coronary heart disease': chd,
            'HGB': hgb,
            'hospitalization': hosp,
            'renal dysfunction': renal
        }
        
        df_raw = pd.DataFrame([inputs])[cols]
        
        try:
            # 预处理
            if imputer:
                X_proc = imputer.transform(df_raw)
                X_scl = scaler.transform(X_proc)
            else:
                X_scl = scaler.transform(df_raw)
            
            # 预测
            prob = model.predict_proba(X_scl)[:, 1][0]
            risk_label = "High Risk" if prob >= THRESHOLD else "Low Risk"
            
            # 存入数据库
            db.add_record(inputs, prob, risk_label)
            
        except Exception as e:
            st.error(f"Computation Error: {e}")
            st.stop()

        st.divider()
        st.subheader("Assessment Results")
        
        res_c1, res_c2 = st.columns([1, 1])
        
        # 结果与 SHAP
        with res_c1:
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Mortality Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': gauge_color}, 
                    'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': THRESHOLD*100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # SHAP 计算与清洗
        sv_clean = [0]*7 # 默认值
        with res_c2:
            st.markdown("**Feature Contribution Analysis (SHAP)**")
            with st.spinner("Analyzing..."):
                try:
                    background = shap.kmeans(scaler.mean_.reshape(1, -1), 1) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_scl, nsamples=50)
                    
                    # === 关键修复：确保 sv 是纯数值列表，不是 array ===
                    if isinstance(shap_values, list): 
                        raw_sv = shap_values[1][0] # 分类模型通常返回 list
                    else:
                        if len(shap_values.shape) == 3: # (1, 7, 2) 情况
                            raw_sv = shap_values[0][:, 1]
                        else:
                            raw_sv = shap_values[0] # (1, 7) 情况

                    # 强制转为 float 列表，防止 sort 报错
                    sv_clean = []
                    for v in raw_sv:
                        if isinstance(v, (np.ndarray, list)):
                            sv_clean.append(float(v.item()) if hasattr(v, 'item') else float(v[0]))
                        else:
                            sv_clean.append(float(v))
                    
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

                    # 画图
                    exp = shap.Explanation(
                        values=np.array(sv_clean), 
                        base_values=base_val, 
                        data=df_raw.iloc[0].values, 
                        feature_names=cols
                    )
                    
                    fig_shap, ax = plt.subplots(figsize=(5, 4))
                    shap.plots.waterfall(exp, max_display=7, show=False)
                    st.pyplot(fig_shap, bbox_inches='tight')
                    plt.clf()
                except Exception as shap_err:
                    st.warning(f"SHAP visualization unavailable: {shap_err}")

        st.divider()
        # 生成文字报告
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv_clean, cols, 0.5)
        full_report = nlg.generate_full_report()
        
        with st.expander("Clinical Report (Text)", expanded=True):
            st.markdown(full_report)
        
        # PDF 下载
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_buffer = io.BytesIO()
        pdf_engine = PDFReportEngine(
            buffer=pdf_buffer,
            patient_data=inputs,
            predict_result={'prob': prob, 'threshold': THRESHOLD, 'risk_label': risk_label},
            nlg_report=full_report
        )
        
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        time_str = beijing_time.strftime("%Y%m%d_%H%M")
        
        st.download_button(
            label="Download PDF Report",
            data=pdf_engine.generate(),
            file_name=f"ATBAD_Report_{inputs['age']}_{time_str}.pdf",
            mime="application/pdf"
        )

# ----------------- PAGE 2: 批量处理 -----------------
elif page == "Batch Analysis":
    st.title("Batch Cohort Analysis")
    st.markdown("Upload dataset for batch risk stratification.")

    with st.expander("Data Formatting Requirements"):
        st.markdown("""
        **Required Columns (Case Sensitive):** `age`, `HR`, `BUN`, `coronary heart disease`, `HGB`, `hospitalization`, `renal dysfunction`
        """)
        
        template_df = pd.DataFrame(columns=[
            'ID', 'age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction'
        ])
        template_df.loc[0] = ['Test_01', 65, 80, 7.5, 0, 130, 10, 0]
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button("Download CSV Template", template_csv, "ATBAD_Batch_Template.csv", "text/csv")

    st.divider()
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=['xlsx', 'csv'])
    
    if uploaded_file:
        processor = BatchProcessor(model, scaler, imputer)
        try:
            if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file)
            else: df_upload = pd.read_excel(uploaded_file)
            
            st.write("Data Preview:", df_upload.head(3))
            
            if st.button("Start Processing"):
                res_df, error = processor.process_data(df_upload)
                if error:
                    st.error(error)
                else:
                    st.success(f"Successfully processed {len(res_df)} records.")
                    st.dataframe(res_df.head())
                    st.download_button("Download Results (.xlsx)", processor.convert_to_excel(res_df), "atbad_batch_results.xlsx")
        except Exception as e:
            st.error(f"File Error: {e}")

# ----------------- PAGE 3: 看板 -----------------
elif page == "Clinical Dashboard":
    st.title("Clinical Data Dashboard")
    analytics = AnalyticsEngine(db)
    df_hist = analytics.get_data()
    
    if df_hist.empty:
        st.info("No historical data available. Please perform assessments first.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Patients", len(df_hist))
        k2.metric("High Risk Proportion", f"{len(df_hist[df_hist['risk_label']=='High Risk']) / len(df_hist):.1%}")
        k3.metric("Avg Predicted Probability", f"{df_hist['risk_prob'].mean():.1%}")
        st.divider()
        st.plotly_chart(analytics.plot_risk_distribution(), use_container_width=True)

# ----------------- PAGE 4: 项目介绍 (原版) -----------------
elif page == "Project Introduction":
    st.title("Machine learning predictive model for three-year mortality in Acute Type B Aortic Dissection (ATBAD)")
    
    st.markdown("""
    ### Abstract & Objective
    **Objective:** To develop accurate machine learning models for predicting three-year mortality in patients with Acute Type B Aortic Dissection (ATBAD), addressing a critical clinical need for improved risk stratification.
    
    **Background:** ATBAD is a life-threatening cardiovascular emergency. While short-term outcomes have improved with TEVAR, long-term mortality remains significant. Identifying high-risk patients is essential for optimizing surveillance and management strategies.
    
    **Methods:** This study enrolled patients with ATBAD from Yichang Central People's Hospital. Comprehensive clinical features including demographics, vital signs, laboratory markers, and comorbidities were analyzed. 
    
    A **Support Vector Machine (SVM)** classifier was identified as the optimal model, demonstrating superior performance compared to Logistic Regression, Random Forest, and GBM in the validation cohort.
    
    ### Key Predictors
    The model integrates the following key clinical variables:
    * **Age:** Patient age at admission.
    * **Heart Rate (HR):** Admission heart rate (bpm).
    * **BUN:** Blood Urea Nitrogen levels (mmol/L).
    * **Coronary Heart Disease:** History of CHD.
    * **Hemoglobin (HGB):** Admission hemoglobin levels (g/L).
    * **Hospitalization:** Length of hospital stay (days).
    * **Renal Dysfunction:** History of renal impairment.

    ---
    *Disclaimer: This web-based calculator is intended for research and educational purposes only. It is not a substitute for professional clinical judgment or diagnosis.*
    """)

    # 说明书下载
    st.markdown("### User Manual")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    manual_path = os.path.join(BASE_DIR, "assets", "ATBAD_User_Manual.docx")
    
    if os.path.exists(manual_path):
        with open(manual_path, "rb") as f:
            st.download_button(
                label="Download User Manual (.docx)",
                data=f,
                file_name="ATBAD_User_Manual.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# --- 页脚 ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; font-family: "Times New Roman", serif;'>
    Copyright &copy; 2026 Yichang Central People's Hospital. All Rights Reserved.<br>
    Powered by Department of Medical Informatics.
</div>
""", unsafe_allow_html=True)
