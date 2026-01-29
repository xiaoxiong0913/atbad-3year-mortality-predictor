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
    page_title="ATBAD Mortality Risk Predictor",
    page_icon="❤️",
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
            .protocol-card { padding: 15px; border-radius: 8px; margin-bottom: 15px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .info-card { border-left: 5px solid #17a2b8; }
        </style>
        """, unsafe_allow_html=True)

local_css("assets/style.css")

# ================= 3. 资源加载 (SVM 版) =================
@st.cache_resource
def load_system():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    try:
        # 加载 SVM 模型和 Scaler
        with open(os.path.join(ASSETS_DIR, "svm_model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        
        # ATBAD 项目可能没有 imputer，如果没有就不加载
        imputer = None
        if os.path.exists(os.path.join(ASSETS_DIR, "imputer.pkl")):
            with open(os.path.join(ASSETS_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
            
        return model, scaler, imputer
    except Exception as e:
        st.error(f"System Error: Failed to load core assets. {e}")
        return None, None, None

model, scaler, imputer = load_system()
db = PatientDatabase()

# ATBAD 模型通常阈值默认为 0.5，如果有特定 cutoff 请在此修改
THRESHOLD = 0.5 

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    st.title("❤️ ATBAD Predictor")
    st.caption("ver 3.0.1 | SVM Powered")
    st.markdown("---")
    
    page = st.radio(
        "System Navigation", 
        ["Individual Assessment", "Batch Analysis", "Clinical Dashboard", "System Documentation"],
        index=0
    )
    
    st.markdown("---")
    if model:
        st.success("✅ SVM Model Online")
        st.info("✅ Database Connected")
    else:
        st.error("❌ System Offline")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 单例预测 (7 Variables) -----------------
if page == "Individual Assessment":
    st.title("🏥 Individual Risk Assessment")
    
    with st.container():
        st.markdown("<div class='protocol-card info-card'><b>Protocol Note:</b> Evaluates 3-year mortality risk for Acute Type B Aortic Dissection patients.</div>", unsafe_allow_html=True)

    with st.form("input_form_atbad"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Demographics & Vitals")
            age = st.number_input("Age (years)", 20, 100, 60, key="input_age")
            hr = st.number_input("Heart Rate (bpm)", 30, 180, 80, key="input_hr")
            hosp = st.number_input("Hospitalization (days)", 1, 100, 10, key="input_hosp")
            
            st.markdown("#### Comorbidities")
            chd = st.selectbox("Coronary Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", key="input_chd")
            
        with col2:
            st.markdown("#### Laboratory Markers")
            bun = st.number_input("BUN", 1.0, 50.0, 7.0, 0.1, key="input_bun")
            hgb = st.number_input("Hemoglobin", 50, 200, 130, key="input_hgb")
            
            st.markdown("#### Renal Status")
            renal = st.selectbox("Renal Dysfunction", [0, 1], format_func=lambda x: "Yes" if x==1 else "No", key="input_renal")
        
        # 底部单位说明
        st.info("ℹ️ Units Reference: BUN in `mmol/L` | Hemoglobin in `g/L`")
        
        submitted = st.form_submit_button("🚀 Run Risk Prediction")

    if submitted and model:
        # 构造输入字典 (Key 必须与 features.txt 一致)
        inputs = {
            'age': age,
            'HR': hr,
            'BUN': bun,
            'coronary heart disease': chd,
            'HGB': hgb,
            'hospitalization': hosp,
            'renal dysfunction': renal
        }
        
        # 转换为 DataFrame (注意列顺序)
        cols = ['age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction']
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
        res_c1, res_c2 = st.columns([1, 1])
        
        with res_c1:
            # 仪表盘
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Mortality Risk</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}, 'threshold': {'line': {'color': "red"}, 'value': THRESHOLD*100}}
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with res_c2:
            # SHAP 解释 (使用 KernelExplainer 兼容 SVM)
            st.subheader("🔍 Feature Contribution")
            with st.spinner("Calculating SHAP values..."):
                try:
                    # 使用 KMeans 汇总背景数据加速计算
                    background = shap.kmeans(scaler.mean_.reshape(1, -1), 1) 
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    
                    # 计算当前样本 SHAP
                    shap_values = explainer.shap_values(X_scl, nsamples=50)
                    
                    # 兼容性提取
                    if isinstance(shap_values, list): sv = shap_values[1][0]
                    else: sv = shap_values[0] # 部分 SVM 实现返回结构不同
                    
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

                    exp = shap.Explanation(
                        values=sv, 
                        base_values=base_val, 
                        data=df_raw.iloc[0].values, 
                        feature_names=cols
                    )
                    
                    fig_shap, ax = plt.subplots(figsize=(5, 4))
                    shap.plots.waterfall(exp, max_display=7, show=False)
                    st.pyplot(fig_shap, bbox_inches='tight')
                    plt.clf()
                except Exception as shap_err:
                    st.warning(f"SHAP visualization unavailable for this model type: {shap_err}")
                    sv = [0]*7 # 兜底

        st.markdown("---")
        # 生成文字报告
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv, cols, 0.5)
        full_report = nlg.generate_full_report()
        
        with st.expander("📄 View AI Clinical Report", expanded=True):
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
        
        col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
        with col_down2:
            st.download_button(
                label="📥 Download Official PDF Report",
                data=pdf_engine.generate(),
                file_name=f"ATBAD_Report_{inputs['age']}_{time_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

# ----------------- PAGE 2: 批量处理 -----------------
elif page == "Batch Analysis":
    st.title("📊 Batch Cohort Analysis")
    st.markdown("Upload Excel/CSV to screen multiple patients.")

    with st.expander("📋 Data Template", expanded=True):
        st.markdown("""
        **Required Columns:** `age`, `HR`, `BUN`, `coronary heart disease` (0/1), `HGB`, `hospitalization`, `renal dysfunction` (0/1)
        """)
        # 生成 ATBAD 专用模板
        template_df = pd.DataFrame(columns=[
            'ID', 'age', 'HR', 'BUN', 'coronary heart disease', 'HGB', 'hospitalization', 'renal dysfunction'
        ])
        template_df.loc[0] = ['Test_01', 65, 80, 7.5, 0, 130, 10, 0]
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        
        st.download_button("📥 Download Template", template_csv, "ATBAD_Template.csv", "text/csv")

    st.divider()
    uploaded_file = st.file_uploader("Upload Data", type=['xlsx', 'csv'])
    
    if uploaded_file:
        processor = BatchProcessor(model, scaler, imputer)
        try:
            if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file)
            else: df_upload = pd.read_excel(uploaded_file)
            
            st.write("Preview:", df_upload.head(3))
            
            if st.button("🚀 Start Processing", type="primary"):
                res_df, error = processor.process_data(df_upload)
                if error:
                    st.error(error)
                else:
                    st.success(f"Processed {len(res_df)} records")
                    st.dataframe(res_df.head())
                    st.download_button("Download Results (Excel)", processor.convert_to_excel(res_df), "atbad_results.xlsx")
        except Exception as e:
            st.error(f"Error: {e}")

# ----------------- PAGE 3: 看板 -----------------
elif page == "Clinical Dashboard":
    st.title("📈 Clinical Dashboard")
    analytics = AnalyticsEngine(db)
    df_hist = analytics.get_data()
    
    if df_hist.empty:
        st.info("No data yet. Run some predictions first.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Cases", len(df_hist))
        k2.metric("High Risk %", f"{len(df_hist[df_hist['risk_label']=='High Risk']) / len(df_hist):.1%}")
        k3.metric("Avg Probability", f"{df_hist['risk_prob'].mean():.1%}")
        st.divider()
        st.plotly_chart(analytics.plot_risk_distribution(), use_container_width=True)

# ----------------- PAGE 4: 文档 (移植旧版 Intro) -----------------
elif page == "System Documentation":
    st.title("ℹ️ About the Model")
    
    # === 移植自旧版 streamlit_app.py 的 Introduction ===
    st.markdown("""
    ### Machine learning predictive model for three-year mortality in Acute Type B Aortic Dissection (ATBAD)
    
    **Background**
    Acute type B aortic dissection (ATBAD) is a life-threatening cardiovascular emergency with high mortality rates. 
    Identifying high-risk patients early is crucial for timely intervention and improved outcomes. 
    While several risk scores exist, they often lack precision for long-term prognosis.
    
    **Objective**
    To develop an accurate machine learning model for predicting **three-year mortality** in patients with ATBAD, 
    addressing the critical clinical need for improved risk stratification.
    
    **Methods**
    This tool utilizes a **Support Vector Machine (SVM)** classifier, which demonstrated superior performance 
    (AUC > 0.90) compared to Logistic Regression and other models in our validation cohort.
    
    **Key Predictors**
    The model integrates 7 key clinical variables:
    1. **Age**: Older age correlates with higher vascular fragility.
    2. **Heart Rate (HR)**: Elevated HR indicates hemodynamic stress.
    3. **BUN**: Renal impairment marker.
    4. **Hemoglobin (HGB)**: Anemia suggests blood loss or chronic illness.
    5. **Hospitalization Days**: Proxy for disease severity/complications.
    6. **Coronary Heart Disease**: Major comorbidity.
    7. **Renal Dysfunction**: Critical prognostic factor.
    
    ---
    *Disclaimer: This tool is intended for research and educational purposes only. It should not replace professional clinical judgment.*
    """)
    
# --- 页脚 ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Deployed by Yichang Central People's Hospital | Powered by AI & Clinical Evidence<br>
    &copy; 2026 Medical Informatics Dept.
</div>
""", unsafe_allow_html=True)
