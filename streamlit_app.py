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
        pass
    
    st.markdown("""
    <style>
        /* 全局字体优化 */
        html, body, [class*="css"] {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 18px; 
        }
        /* Overview 卡片 (放在右侧时) */
        .overview-card { 
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 8px; 
            border-left: 6px solid #007bff; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            height: 100%; /* 尝试撑满高度 */
        }
        /* 按钮样式：全宽 */
        .stButton>button {
            width: 100%;
            height: 3.5em;
            font-weight: bold;
            font-size: 1.2rem;
            margin-top: 25px; /* 增加顶部间距，与表单分开 */
        }
        /* 输入框标签 */
        .stNumberInput label, .stSelectbox label {
            font-weight: 600;
            font-size: 1rem;
        }
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

# === 核心截断值 ===
THRESHOLD = 0.207 

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

# ----------------- PAGE 1: 风险评估 -----------------
if page == "Risk Assessment":
    st.title("Individual Risk Assessment")
    st.markdown("---")

    # === 布局核心：左侧 (2) 表单，右侧 (1) 简介 ===
    col_left, col_right = st.columns([2, 1])
    
    # --- 左侧：输入表单 ---
    with col_left:
        st.markdown("##### Patient Clinical Data")
        with st.form("input_form_atbad"):
            # 内部再分3列
            c1, c2, c3 = st.columns(3)
            
            # 第一列
            with c1:
                age = st.number_input("Age (years)", 20, 100, 60)
                hr = st.number_input("Heart Rate (bpm)", 30, 180, 80)
                hosp = st.number_input("Hospitalization (days)", 1, 100, 10)
                
            # 第二列
            with c2:
                bun = st.number_input("BUN (mmol/L)", 0.1, 100.0, 7.0, 0.1)
                hgb = st.number_input("Hemoglobin (g/L)", 30, 250, 130)
                # 占位符，保持对齐
                st.write("") 
                
            # 第三列 (单独放 CHD 和 Renal)
            with c3:
                chd = st.selectbox("Coronary Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                renal = st.selectbox("Renal Dysfunction", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            
            # 按钮 (在表单内底部)
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("CALCULATE RISK", type="primary")

    # --- 右侧：Model Overview 卡片 ---
    with col_right:
        st.markdown(f"""
        <div class='overview-card'>
            <h3 style='margin-bottom:15px; margin-top:0; font-size:1.4rem;'>3-Year Mortality Prediction</h3>
            <h4 style='margin-bottom:10px; color:#555; font-size:1.1rem;'>Model Overview</h4>
            <p style='font-size:15px; line-height:1.6;'>
                This predictive tool uses an <b>SVM machine learning model</b> to estimate 3-year mortality risk in patients with acute Type B aortic dissection.<br><br>
                <b>Performance:</b><br>
                - AUC: <b>0.94</b><br>
                - Accuracy: <b>88.8%</b><br>
                - Risk Threshold: <b>{THRESHOLD}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

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
        
        # === 高危提示 (如果触发) ===
        if prob >= THRESHOLD:
            st.markdown(f"""
            <div style='background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
                <p style='color: #dc3545; font-weight: bold; font-size: 16px; margin: 0;'>
                    ⚠️ High Risk Alert: This patient is in the high-risk group for 3-year mortality. 
                    Consider closer surveillance (CTA every 3-6 months) and aggressive risk factor modification.
                </p>
            </div>
            """, unsafe_allow_html=True)

        res_c1, res_c2 = st.columns([1, 1])
        
        with res_c1:
            # 仪表盘
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Mortality Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': gauge_color}, 
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': THRESHOLD*100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # === 棒棒糖图 (Lollipop Chart) ===
        sv_clean = np.zeros(7)
        with res_c2:
            st.markdown("**Feature Impact Analysis (SHAP)**")
            with st.spinner("Analyzing..."):
                try:
                    background = shap.kmeans(scaler.mean_.reshape(1, -1), 1) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    shap_values = explainer.shap_values(X_scl, nsamples=50)
                    
                    flat_vals = np.array(shap_values).flatten()
                    if len(flat_vals) == 14: sv_clean = flat_vals[7:] 
                    elif len(flat_vals) == 7: sv_clean = flat_vals
                    else: sv_clean = flat_vals[:7] if len(flat_vals) >= 7 else np.zeros(7)
                    
                    sv_clean = np.array([float(x) for x in sv_clean])

                    df_shap = pd.DataFrame({
                        'Feature': [c.split('(')[0] for c in cols],
                        'Impact': sv_clean
                    })
                    df_shap['Abs'] = df_shap['Impact'].abs()
                    df_shap = df_shap.sort_values('Abs', ascending=True)
                    df_shap['Color'] = ['#FF4B4B' if x > 0 else '#1F77B4' for x in df_shap['Impact']]

                    fig_lolly = go.Figure()
                    for index, row in df_shap.iterrows():
                        fig_lolly.add_shape(
                            type='line',
                            x0=0, y0=row['Feature'],
                            x1=row['Impact'], y1=row['Feature'],
                            line=dict(color='gray', width=1)
                        )

                    fig_lolly.add_trace(go.Scatter(
                        x=df_shap['Impact'],
                        y=df_shap['Feature'],
                        mode='markers',
                        marker=dict(color=df_shap['Color'], size=12),
                        name='Impact',
                        showlegend=False,
                        hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>'
                    ))

                    fig_lolly.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=20, b=20),
                        xaxis=dict(title="Impact on Risk Probability", zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                        yaxis=dict(showgrid=False),
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig_lolly, use_container_width=True)

                except Exception as shap_err:
                    st.warning(f"Feature Analysis Unavailable: {shap_err}")
                    sv_clean = [0.0] * 7

        st.divider()
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv_clean.tolist(), cols, 0.5)
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
                if 'Mortality_Risk_Prob' in res_df.columns:
                    res_df['Risk_Level'] = res_df['Mortality_Risk_Prob'].apply(lambda x: "High Risk" if x >= THRESHOLD else "Low Risk")
                
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
