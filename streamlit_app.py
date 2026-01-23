import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go
import datetime # 引入时间模块

# ================= 1. 引用自定义模块 =================
from modules.database import PatientDatabase
from modules.nlg_generator import ClinicalReportGenerator
from modules.pdf_report import PDFReportEngine
from modules.batch_processor import BatchProcessor
from modules.analytics import AnalyticsEngine

# ================= 2. 系统初始化与配置 =================
st.set_page_config(
    page_title="DR-MACE Clinical Decision Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown("""
        <style>
            .protocol-card { padding: 15px; border-radius: 8px; margin-bottom: 15px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .critical-card { border-left: 5px solid #dc3545; }
            .safe-card { border-left: 5px solid #28a745; }
        </style>
        """, unsafe_allow_html=True)

local_css("assets/style.css")

# ================= 3. 资源加载 =================
@st.cache_resource
def load_system():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    try:
        with open(os.path.join(ASSETS_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
        return model, scaler, imputer
    except Exception as e:
        st.error(f"System Error: {e}")
        return None, None, None

model, scaler, imputer = load_system()
db = PatientDatabase()
THRESHOLD = 0.193

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    st.title("🩺 DR-MACE System")
    st.caption("ver 2.0.2 | Enterprise Edition")
    st.markdown("---")
    page = st.radio(
        "System Navigation", 
        ["Individual Assessment", "Batch Cohort Analysis", "Clinical Dashboard", "System Documentation"],
        index=0
    )
    st.markdown("---")
    if model:
        st.success("✅ Model Online")
        st.info("✅ Database Connected")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 单例预测 -----------------
if page == "Individual Assessment":
    st.title("🏥 Individual Patient Assessment")
    
    with st.container():
        st.markdown("<div class='protocol-card info-card'><b>Protocol Note:</b> Ensure all lab values are from within the last 30 days.</div>", unsafe_allow_html=True)

    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Demographics & Vitals")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 130)
            t_wave = st.selectbox("ECG: T-Wave Abnormalities", [0, 1], format_func=lambda x: "Present" if x==1 else "Absent")
        with col2:
            st.markdown("#### Laboratory & Meds")
            hgb = st.number_input("Hemoglobin (g/L)", 30, 250, 135)
            bun = st.number_input("BUN (mmol/L)", 0.0, 100.0, 7.0, 0.1)
            statins = st.selectbox("Statin Therapy", [0, 1], format_func=lambda x: "On Therapy" if x==1 else "Naive/None")
        
        submitted = st.form_submit_button("🚀 Run Risk Assessment")

    if submitted and model:
        inputs = {
            'BUN(mmol/L)': bun, 'SBP(mmHg)': sbp, 'HGB(g/L)': hgb,
            'T wave  abnormalities': t_wave, 'Statins': statins, 'Gender': gender
        }
        cols = ['BUN(mmol/L)', 'SBP(mmHg)', 'HGB(g/L)', 'T wave  abnormalities', 'Statins']
        df_raw = pd.DataFrame([inputs]).drop(columns=['Gender'])[cols]
        
        try:
            X_imp = imputer.transform(df_raw)
            X_scl = scaler.transform(X_imp)
            df_scl = pd.DataFrame(X_scl, columns=cols)
            prob = model.predict_proba(df_scl)[:, 1][0]
            risk_label = "High Risk" if prob >= THRESHOLD else "Low Risk"
            db.add_record(inputs, prob, risk_label)
        except Exception as e:
            st.error(f"Computation Error: {e}")
            st.stop()

        st.divider()
        res_c1, res_c2 = st.columns([1, 1])
        with res_c1:
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Risk Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': gauge_color}, 'threshold': {'line': {'color': "red"}, 'value': THRESHOLD*100}}
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with res_c2:
            st.subheader("🔍 Factor Contribution")
            with st.spinner("Analyzing..."):
                background = pd.DataFrame(np.zeros((1, 5)), columns=cols)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(df_scl, nsamples=100)
                
                if isinstance(shap_values, list): sv = shap_values[1][0]
                elif len(np.array(shap_values).shape) == 3: sv = shap_values[0][:, 1]
                else: sv = shap_values[0]

                ev = explainer.expected_value
                if isinstance(ev, np.ndarray) and ev.size > 1: base_val = ev[1]
                elif isinstance(ev, list): base_val = ev[1]
                else: base_val = ev
                if hasattr(base_val, 'item'): base_val = base_val.item()
                
                exp = shap.Explanation(
                    values=sv, base_values=base_val, data=df_scl.iloc[0].values, 
                    feature_names=[c.split('(')[0] for c in cols]
                )
                fig_shap, ax = plt.subplots(figsize=(5, 4))
                shap.plots.waterfall(exp, max_display=5, show=False)
                st.pyplot(fig_shap, bbox_inches='tight')
                plt.clf()

        st.markdown("---")
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv, cols, base_val)
        full_report = nlg.generate_full_report()
        
        with st.expander("📄 View AI Clinical Report (Full Text)", expanded=True):
            st.markdown(full_report)
        
        # --- 关键修改：布局调整与时区修正 ---
        st.markdown("<br><br>", unsafe_allow_html=True) # 增加一些垂直间距
        
        # 1. 生成 PDF
        pdf_buffer = io.BytesIO()
        pdf_engine = PDFReportEngine(
            buffer=pdf_buffer,
            patient_data=inputs,
            predict_result={'prob': prob, 'threshold': THRESHOLD, 'risk_label': risk_label},
            nlg_report=full_report
        )
        
        # 2. 获取北京时间字符串用于文件名
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        time_str = beijing_time.strftime("%Y%m%d_%H%M")
        
        # 3. 放在最底部居中位置
        col_down1, col_down2, col_down3 = st.columns([1, 2, 1])
        with col_down2:
            st.download_button(
                label="📥 Download Official PDF Report",
                data=pdf_engine.generate(),
                file_name=f"Report_{inputs['SBP(mmHg)']}_{time_str}.pdf", # 文件名带上北京时间
                mime="application/pdf",
                use_container_width=True,
                type="primary" # 突出显示为主按钮样式
            )
        # --------------------------------

# ----------------- PAGE 2: 批量处理 -----------------
elif page == "Batch Cohort Analysis":
    st.title("📊 Retrospective Cohort Analysis")
    uploaded_file = st.file_uploader("Upload Dataset", type=['xlsx', 'csv'])
    if uploaded_file:
        processor = BatchProcessor(model, scaler, imputer)
        if uploaded_file.name.endswith('.csv'): df_upload = pd.read_csv(uploaded_file)
        else: df_upload = pd.read_excel(uploaded_file)
        st.write("Preview:", df_upload.head(3))
        if st.button("Start Batch Processing"):
            res_df, error = processor.process_data(df_upload)
            if error: st.error(error)
            else:
                st.success("Done!")
                st.dataframe(res_df.head())
                st.download_button("Download CSV", processor.convert_to_csv(res_df), "batch_res.csv")

# ----------------- PAGE 3: 统计看板 -----------------
elif page == "Clinical Dashboard":
    st.title("📈 Clinical Data Dashboard")
    analytics = AnalyticsEngine(db)
    df_hist = analytics.get_data()
    if df_hist.empty: st.info("No data available.")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Patients", len(df_hist))
        k2.metric("High Risk", f"{len(df_hist[df_hist['risk_label']=='High Risk'])}")
        k3.metric("Avg Prob", f"{df_hist['risk_prob'].mean():.1%}")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(analytics.plot_risk_distribution(), use_container_width=True)
        with c2: st.plotly_chart(analytics.plot_gender_stats(), use_container_width=True)

# ----------------- PAGE 4: 文档 -----------------
elif page == "System Documentation":
    st.markdown("### System Specifications")
    st.info("Architecture: Modular MVC (Streamlit + SQLite + ReportLab)")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Deployed by Yichang Central People's Hospital | Powered by AI & Clinical Evidence<br>
    &copy; 2026 Medical Informatics Dept.
</div>
""", unsafe_allow_html=True)
