import os
import streamlit as st

# === 调试代码：查看服务器上的文件结构 ===
st.write("Current Working Directory:", os.getcwd())
st.write("Files in root:", os.listdir('.'))
if os.path.exists('modules'):
    st.write("Files in modules:", os.listdir('modules'))
else:
    st.error("❌ 'modules' folder NOT found!")
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt
import shap
import plotly.graph_objects as go

# ================= 1. 引用自定义模块 (模块化架构) =================
# 这里的引用依赖于您已经建立了 modules 文件夹
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

# 加载外部 CSS (从 assets/style.css)
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # 如果本地没找到文件，为了防止报错，注入默认样式
        st.warning(f"Note: Style file '{file_name}' not found. Using default styles.")
        st.markdown("""
        <style>
            .protocol-card { padding: 15px; border-radius: 8px; margin-bottom: 15px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .critical-card { border-left: 5px solid #dc3545; }
            .safe-card { border-left: 5px solid #28a745; }
        </style>
        """, unsafe_allow_html=True)

# 指向 assets 文件夹加载 CSS
local_css("assets/style.css")

# ================= 3. 资源加载 (适配 GitHub 结构) =================
@st.cache_resource
def load_system():
    # 获取当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 设定资源文件夹路径
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")
    
    try:
        # 加载模型文件
        with open(os.path.join(ASSETS_DIR, "Naive_Bayes_Model.pkl"), 'rb') as f: model = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "scaler.pkl"), 'rb') as f: scaler = pickle.load(f)
        with open(os.path.join(ASSETS_DIR, "imputer.pkl"), 'rb') as f: imputer = pickle.load(f)
        return model, scaler, imputer
    except FileNotFoundError as e:
        st.error(f"System Error: Critical resource missing in {ASSETS_DIR}. Details: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"System Error: Failed to load pipeline. {e}")
        return None, None, None

model, scaler, imputer = load_system()
db = PatientDatabase() # 初始化数据库连接

# 全局阈值 (基于 Manuscript)
THRESHOLD = 0.193

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    st.title("🩺 DR-MACE System")
    st.caption("ver 2.0.0 | Enterprise Edition")
    st.markdown("---")
    
    page = st.radio(
        "System Navigation", 
        ["Individual Assessment", "Batch Cohort Analysis", "Clinical Dashboard", "System Documentation"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Session Info**")
    if model:
        st.success("✅ Model Online")
        st.info("✅ Database Connected")
    else:
        st.error("❌ System Offline")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 单例预测 (核心功能) -----------------
if page == "Individual Assessment":
    st.title("🏥 Individual Patient Assessment")
    st.markdown("Enter patient clinical parameters below to generate a real-time risk stratification report.")
    
    with st.container():
        st.markdown("<div class='protocol-card info-card'><b>Protocol Note:</b> Ensure all lab values are from within the last 30 days.</div>", unsafe_allow_html=True)

    # 表单输入
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Demographics & Vitals")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            sbp = st.number_input("Systolic BP (mmHg)", 50, 250, 130, help="Target: <140 mmHg")
            t_wave = st.selectbox("ECG: T-Wave Abnormalities", [0, 1], format_func=lambda x: "Present (Pathological)" if x==1 else "Absent")
            
        with col2:
            st.markdown("#### Laboratory & Meds")
            hgb_ref = "130-175" if gender == "Male" else "120-155"
            hgb = st.number_input(f"Hemoglobin (g/L) [Ref: {hgb_ref}]", 30, 250, 135)
            bun = st.number_input("BUN (mmol/L) [Ref: 2.8-7.1]", 0.0, 100.0, 7.0, 0.1)
            statins = st.selectbox("Statin Therapy", [0, 1], format_func=lambda x: "On Therapy" if x==1 else "Naive/None")
        
        submitted = st.form_submit_button("🚀 Run Risk Assessment")

    if submitted and model:
        # 1. 数据封装
        inputs = {
            'BUN(mmol/L)': bun, 'SBP(mmHg)': sbp, 'HGB(g/L)': hgb,
            'T wave  abnormalities': t_wave, 'Statins': statins, 'Gender': gender
        }
        
        # 2. 预处理
        cols = ['BUN(mmol/L)', 'SBP(mmHg)', 'HGB(g/L)', 'T wave  abnormalities', 'Statins']
        df_raw = pd.DataFrame([inputs]).drop(columns=['Gender'])[cols]
        
        try:
            X_imp = imputer.transform(df_raw)
            X_scl = scaler.transform(X_imp)
            df_scl = pd.DataFrame(X_scl, columns=cols)
            
            # 3. 预测
            prob = model.predict_proba(df_scl)[:, 1][0]
            risk_label = "High Risk" if prob >= THRESHOLD else "Low Risk"
            
            # 4. 存入数据库
            db.add_record(inputs, prob, risk_label)
            
        except Exception as e:
            st.error(f"Computation Error: {e}")
            st.stop()

        # --- 结果展示区 ---
        st.divider()
        res_c1, res_c2 = st.columns([1, 1])
        
        # 左侧：仪表盘
        with res_c1:
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>3-Year MACE Probability</b><br><span style='color:gray;font-size:0.8em'>{risk_label}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': gauge_color},
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': THRESHOLD*100}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # 右侧：SHAP 瀑布图 (静态图修复版)
        with res_c2:
            st.subheader("🔍 Factor Contribution")
            with st.spinner("Analyzing..."):
                background = pd.DataFrame(np.zeros((1, 5)), columns=cols)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(df_scl, nsamples=100)
                
                # SHAP 数据提取逻辑 (兼容性修复)
                if isinstance(shap_values, list): sv = shap_values[1][0]
                else: sv = shap_values[0]
                
                base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                if hasattr(base_val, 'item'): base_val = base_val.item()
                
                # 构建解释对象
                exp = shap.Explanation(
                    values=sv, 
                    base_values=base_val, 
                    data=df_scl.iloc[0].values, 
                    feature_names=[c.split('(')[0] for c in cols] # 简化显示名称
                )
                
                # 绘制 Matplotlib 静态图
                fig_shap, ax = plt.subplots(figsize=(5, 4))
                shap.plots.waterfall(exp, max_display=5, show=False)
                st.pyplot(fig_shap, bbox_inches='tight')
                plt.clf() # 清理画布

        # --- 报告生成区 ---
        st.markdown("---")
        
        # 调用 NLG 引擎生成文本
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv, cols, base_val)
        full_report = nlg.generate_full_report()
        
        c_rep1, c_rep2 = st.columns([3, 1])
        with c_rep1:
            with st.expander("📄 View AI-Generated Clinical Report", expanded=True):
                st.markdown(full_report)
        
        with c_rep2:
            st.markdown("<br>", unsafe_allow_html=True) # 占位符
            # 调用 PDF 引擎
            pdf_buffer = io.BytesIO()
            pdf_engine = PDFReportEngine(
                buffer=pdf_buffer,
                patient_data=inputs,
                predict_result={'prob': prob, 'threshold': THRESHOLD, 'risk_label': risk_label},
                nlg_report=full_report
            )
            pdf_bytes = pdf_engine.generate()
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"MACE_Report_{inputs['SBP(mmHg)']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ----------------- PAGE 2: 批量处理 (科研功能) -----------------
elif page == "Batch Cohort Analysis":
    st.title("📊 Retrospective Cohort Analysis")
    st.markdown("Upload a dataset (Excel/CSV) to perform batch inference for research purposes.")
    
    with st.container():
        st.markdown("""
        <div class='protocol-card warning-card'>
            <b>Requirements:</b> File must contain columns: <code>BUN(mmol/L)</code>, <code>SBP(mmHg)</code>, <code>HGB(g/L)</code>, <code>T wave abnormalities</code>, <code>Statins</code>.
        </div>
        """, unsafe_allow_html=True)
        
    uploaded_file = st.file_uploader("Upload Dataset", type=['xlsx', 'csv'])
    
    if uploaded_file:
        processor = BatchProcessor(model, scaler, imputer)
        
        # 读取文件
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file)
        else:
            df_upload = pd.read_excel(uploaded_file)
            
        st.write("Data Preview:", df_upload.head(3))
        
        if st.button("Start Batch Processing"):
            with st.spinner("Processing cohort data..."):
                res_df, error_msg = processor.process_data(df_upload)
                
                if error_msg:
                    st.error(f"Processing Failed: {error_msg}")
                else:
                    st.success(f"Successfully processed {len(res_df)} records.")
                    st.dataframe(res_df.head())
                    
                    # 下载区域
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        csv_data = processor.convert_to_csv(res_df)
                        st.download_button("Download CSV", csv_data, "batch_results.csv", "text/csv")
                    with d_col2:
                        xlsx_data = processor.convert_to_excel(res_df)
                        st.download_button("Download Excel", xlsx_data, "batch_results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------- PAGE 3: 统计看板 (管理功能) -----------------
elif page == "Clinical Dashboard":
    st.title("📈 Clinical Data Dashboard")
    st.markdown("Real-time statistics derived from the patient history database.")
    
    analytics = AnalyticsEngine(db)
    df_hist = analytics.get_data()
    
    if df_hist.empty:
        st.info("No historical data found. Please run individual assessments first.")
    else:
        # KPI 指标
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Patients", len(df_hist))
        kpi2.metric("High Risk Ratio", f"{len(df_hist[df_hist['risk_label']=='High Risk']) / len(df_hist):.1%}")
        kpi3.metric("Avg MACE Prob", f"{df_hist['risk_prob'].mean():.1%}")
        
        st.divider()
        
        # 图表区域
        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(analytics.plot_risk_distribution(), use_container_width=True)
        with chart2:
            st.plotly_chart(analytics.plot_gender_stats(), use_container_width=True)
            
        st.plotly_chart(analytics.plot_temporal_trend(), use_container_width=True)
        
        with st.expander("View Raw Data Log"):
            st.dataframe(df_hist)
            if st.button("Purge Database (Admin)"):
                db.delete_all_records()
                st.rerun()

# ----------------- PAGE 4: 系统文档 -----------------
elif page == "System Documentation":
    st.title("ℹ️ System Specifications")
    st.markdown("""
    ### DR-MACE Risk Stratification System v2.0
    
    #### 1. Software Architecture
    This system is built using a **Modular MVC Architecture** designed for high scalability and maintainability, compliant with software copyright standards.
    
    * **Frontend**: Streamlit (Reactive Web Framework)
    * **Backend Logic**: Scikit-learn (Machine Learning), Pandas (Data Processing)
    * **Persistence**: SQLite3 (Local Relational Database)
    * **Visualization**: Plotly (Interactive), Matplotlib/SHAP (Static Inference)
    * **Reporting**: ReportLab (Vector PDF Generation)
    
    #### 2. Core Algorithm
    * **Model**: Naive Bayes Classifier (GaussianNB)
    * **Validation**: 5-Fold Cross Validation on Multi-center Cohort (N=390)
    * **Performance**: AUC 0.771 | Sensitivity 82% | Specificity 71%
    
    #### 3. Intellectual Property
    * **Developer**: Research Team @ Yichang Central People's Hospital
    * **License**: Proprietary / Research Use Only
    """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Deployed by Yichang Central People's Hospital | Powered by AI & Clinical Evidence<br>
    &copy; 2025 Medical Informatics Dept.
</div>
""", unsafe_allow_html=True)

