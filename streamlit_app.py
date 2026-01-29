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
    page_title="ATBAD Mortality Risk Prediction",
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
            /* 简洁风格 */
            .main-header { font-family: 'Helvetica Neue', sans-serif; font-weight: bold; color: #333; }
            .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
            .stButton>button:hover { background-color: #0056b3; color: white; }
            /* 输入框标签加粗 */
            .stNumberInput label, .stSelectbox label { font-weight: 600; font-size: 0.9rem; }
            /* 右侧卡片样式 */
            .overview-card { background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
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
        st.error(f"System Error: {e}")
        return None, None, None

model, scaler, imputer = load_system()
db = PatientDatabase()

THRESHOLD = 0.5 

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Risk Assessment", "Batch Analysis", "Clinical Dashboard", "Project Introduction"])
    st.markdown("---")
    if model:
        st.success("System Online")
    else:
        st.error("System Offline")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 单例预测 (重构版) -----------------
if page == "Risk Assessment":
    st.title("Individual Risk Assessment")
    st.markdown("---")

    # 布局：左侧表单 (2)，右侧说明 (1)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("input_form_atbad"):
            st.markdown("##### Patient Clinical Data")
            
            # 第一行：3个基础指标
            r1c1, r1c2, r1c3 = st.columns(3)
            with r1c1: age = st.number_input("Age (years)", 20, 100, 60)
            with r1c2: hr = st.number_input("Heart Rate (bpm)", 30, 180, 80)
            with r1c3: hgb = st.number_input("Hemoglobin (g/L)", 30, 250, 130)
            
            # 第二行：3个进阶指标
            r2c1, r2c2, r2c3 = st.columns(3)
            with r2c1: bun = st.number_input("BUN (mmol/L)", 0.1, 100.0, 7.0, 0.1)
            with r2c2: hosp = st.number_input("Hospitalization (days)", 1, 100, 10)
            with r2c3: chd = st.selectbox("Coronary Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            # 第三行：1个指标
            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1: renal = st.selectbox("Renal Dysfunction", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("CALCULATE RISK")

    with col2:
        # === 严格只放您要求的代码 ===
        st.markdown("<div class='overview-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:10px; margin-top:0;'>3-Year Mortality Prediction for Acute Type B Aortic Dissection</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:10px;'>Model Overview</h4>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:14px;'>
        This predictive tool uses an SVM machine learning model to estimate 3-year mortality risk in patients with acute Type B aortic dissection.<br>
        - AUC: <b>0.94</b><br>
        - Accuracy: <b>88.8%</b><br>
        - Risk Threshold: <b>0.207</b>
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted and model:
        # 特征映射
        cols = ['age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction']
        inputs = {'age': age, 'HR': hr, 'BUN': bun, 'coronary heart disease': chd, 
                  'HGB': hgb, 'hospitalization': hosp, 'renal dysfunction': renal}
        
        df_raw = pd.DataFrame([inputs])[cols]
        
        try:
            if imputer:
                X_scl = scaler.transform(imputer.transform(df_raw))
            else:
                X_scl = scaler.transform(df_raw)
            
            prob = model.predict_proba(X_scl)[:, 1][0]
            risk_label = "High Risk" if prob >= THRESHOLD else "Low Risk"
            db.add_record(inputs, prob, risk_label)
            
        except Exception as e:
            st.error(f"Computation Error: {e}")
            st.stop()

        st.divider()
        st.subheader("Prediction Results")
        
        res_c1, res_c2 = st.columns([1, 1])
        
        with res_c1:
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Mortality Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}, 'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': THRESHOLD*100}}
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # SHAP 修复逻辑
        sv_clean = np.zeros(7)
        with res_c2:
            st.markdown("**Feature Contribution (SHAP)**")
            with st.spinner("Analyzing..."):
                try:
                    background = shap.kmeans(scaler.mean_.reshape(1, -1), 1) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_scl, nsamples=50)
                    
                    # === 终极修复：处理所有可能的数组形状 ===
                    # 目标：拿到 positive class 的 (7,) 一维数组
                    if isinstance(shap_values, list):
                        target = shap_values[1] # 二分类取第二个
                    else:
                        target = shap_values

                    # 如果是 (1, 7, 2) -> 取 (0, :, 1) -> (7,)
                    # 如果是 (1, 7) -> 取 (0, :) -> (7,)
                    target = np.array(target)
                    if len(target.shape) == 3: # (samples, features, classes)
                        sv_clean = target[0, :, 1]
                    elif len(target.shape) == 2: # (samples, features)
                        sv_clean = target[0, :]
                    else:
                        sv_clean = target # 已经是 (features,)

                    # 确保是 float 类型
                    sv_clean = sv_clean.astype(float)
                    
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

                    exp = shap.Explanation(
                        values=sv_clean, 
                        base_values=base_val, 
                        data=df_raw.iloc[0].values, 
                        feature_names=cols
                    )
                    
                    fig_shap, ax = plt.subplots(figsize=(5, 4))
                    shap.plots.waterfall(exp, max_display=7, show=False)
                    st.pyplot(fig_shap, bbox_inches='tight')
                    plt.clf()
                except Exception as shap_err:
                    st.warning(f"SHAP Error: {shap_err}")

        st.divider()
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv_clean, cols, 0.5)
        full_report = nlg.generate_full_report()
        
        with st.expander("📄 View Clinical Report", expanded=True):
            st.markdown(full_report)
        
        pdf_buffer = io.BytesIO()
        pdf_engine = PDFReportEngine(pdf_buffer, inputs, {'prob': prob, 'threshold': THRESHOLD, 'risk_label': risk_label}, full_report)
        
        time_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d_%H%M")
        st.download_button("Download PDF Report", pdf_engine.generate(), f"Report_{time_str}.pdf", "application/pdf")

# ----------------- PAGE 2: 批量处理 -----------------
elif page == "Batch Analysis":
    st.title("Batch Cohort Analysis")
    with st.expander("Data Formatting"):
        st.markdown("**Required Columns:** `age`, `HR`, `BUN`, `coronary heart disease`, `HGB`, `hospitalization`, `renal dysfunction`")
        template = pd.DataFrame(columns=['ID', 'age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction'])
        st.download_button("Download Template", template.to_csv(index=False).encode('utf-8'), "Batch_Template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload File", type=['xlsx', 'csv'])
    if uploaded_file:
        processor = BatchProcessor(model, scaler, imputer)
        if st.button("Start Processing"):
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            res_df, err = processor.process_data(df)
            if err: st.error(err)
            else:
                st.success(f"Processed {len(res_df)} records")
                st.dataframe(res_df.head())
                st.download_button("Download Results", processor.convert_to_excel(res_df), "Results.xlsx")

# ----------------- PAGE 3: 看板 -----------------
elif page == "Clinical Dashboard":
    st.title("Clinical Dashboard")
    df_hist = AnalyticsEngine(db).get_data()
    if df_hist.empty: st.info("No data available.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Cases", len(df_hist))
        k2.metric("High Risk %", f"{len(df_hist[df_hist['risk_label']=='High Risk']) / len(df_hist):.1%}")
        k3.metric("Avg Risk", f"{df_hist['risk_prob'].mean():.1%}")
        st.plotly_chart(AnalyticsEngine(db).plot_risk_distribution(), use_container_width=True)

# ----------------- PAGE 4: 介绍 -----------------
elif page == "Project Introduction":
    st.title("ATBAD Mortality Prediction Model")
    st.markdown("""
    ### Abstract
    **Objective:** To develop accurate machine learning models for predicting three-year mortality in patients with Acute Type B Aortic Dissection (ATBAD).
    
    **Methods:** This study enrolled patients from Yichang Central People's Hospital. A **Support Vector Machine (SVM)** classifier was identified as the optimal model (AUC 0.94).
    
    **Key Predictors:** Age, Heart Rate, BUN, Coronary Heart Disease, Hemoglobin, Hospitalization, Renal Dysfunction.
    """)
    
    manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "ATBAD_User_Manual.docx")
    if os.path.exists(manual_path):
        with open(manual_path, "rb") as f:
            st.download_button("Download User Manual", f, "ATBAD_User_Manual.docx")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 0.8em;'>Copyright &copy; 2026 Yichang Central People's Hospital.</div>", unsafe_allow_html=True)
