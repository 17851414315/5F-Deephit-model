import os
os.environ['PYCOX_NO_SAVE'] = '1'  # 禁止pycox尝试保存数据到本地
import streamlit as st
import numpy as np
import torch
import pandas as pd
from pycox.evaluation import EvalSurv
import matplotlib.pyplot as plt
import time

# 设置页面标题和布局
st.set_page_config(
    page_title="5F-Deephit model",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 加载模型 (使用缓存避免重复加载)
@st.cache_resource
def load_model():
    return torch.load("deephit5sig.pt")


# 页面标题和说明
st.title("5F-Deephit model")
st.markdown("""
    <style>
    .big-font {
        font-size:16px !important;
        color: #4a4a4a;
    }
    </style>
    <p class="big-font">Please input the concentration values of five amino acid biomarkers to predict the patient's 30 day and 60 day survival rates</p>
    """, unsafe_allow_html=True)

# 创建输入表单
with st.form("prediction_form"):
    st.subheader("Biomarker concentration")

    col1, col2 = st.columns(2)

    with col1:
        PGA = st.number_input("PGA μmol/L", min_value=0.0, max_value=1000.0, value=10.0, step=0.1, format="%.3f")
        PRO = st.number_input("PRO μmol/L", min_value=0.0, max_value=1000.0, value=11.0, step=0.1, format="%.3f")
        ALA = st.number_input("ALA μmol/L", min_value=0.0, max_value=1000.0, value=12.0, step=0.1, format="%.3f")

    with col2:
        GLY = st.number_input("GLY μmol/L", min_value=0.0, max_value=1000.0, value=13.0, step=0.1, format="%.3f")
        HIS = st.number_input("HIS μmol/L", min_value=0.0, max_value=1000.0, value=14.0, step=0.1, format="%.3f")

    submitted = st.form_submit_button("Survival rate prediction", use_container_width=True)

# 当表单提交时进行预测
if submitted:
    # 数据验证
    if any(val < 0 for val in [PGA, PRO, ALA, GLY, HIS]):
        st.error("Error: Concentration value cannot be negative!")
        st.stop()

    if all(val == 0 for val in [PGA, PRO, ALA, GLY, HIS]):
        st.warning("Warning: All concentration values are zero, please check the input!")
        st.stop()

    # 显示进度条
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 模拟加载过程
    for percent_complete in range(100):
        time.sleep(0.01)  # 模拟计算时间
        progress_bar.progress(percent_complete + 1)
        status_text.text(f"In the process of calculation ... {percent_complete + 1}%")

    try:
        # 准备输入数据
        pdata1 = pd.DataFrame({"dat": (PGA, PRO, ALA, GLY, HIS)})
        pdata2 = pdata1.transpose().values.astype("float32")

        # 进行预测
        model = load_model()
        pred1 = model.interpolate(10).predict_surv_df(pdata2)
        ev1 = EvalSurv(pred1, np.array(1), np.array(1))

        # 计算生存率
        s30 = ev1.surv_at_times(30)
        s30pro = round(s30[0] * 100, 2)
        s60 = ev1.surv_at_times(60)
        s60pro = round(s60[0] * 100, 2)

        # 清除进度条
        progress_bar.empty()
        status_text.empty()

        # 显示结果
        st.success("Prediction completed!")

        # 使用列布局显示结果
        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="30 day survival rate", value=f"{s30pro}%")
            st.metric(label="60 day survival rate", value=f"{s60pro}%")

            # 显示输入值
            st.write("### Review of Input Values")
            input_data = {
                "Biomarkers": ["PGA", "PRO", "ALA", "GLY", "HIS"],
                "Concentration(μmol/L)": [PGA, PRO, ALA, GLY, HIS]
            }
            st.table(input_data)

        with col2:
            # 绘制生存曲线
            st.write("### Survival probability curve")
            fig, ax = plt.subplots(figsize=(8, 4))
            pred1.plot(ax=ax, linewidth=2.5)

            # 添加30天和60天的标记线
            ax.axvline(x=30, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=60, color='blue', linestyle='--', alpha=0.5)
            ax.text(30 + 2, 0.9, '30 day', color='red')
            ax.text(60 + 2, 0.9, '60 day', color='blue')

            ax.set_title("Survival probability over time curve")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Survival Probability")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

            st.pyplot(fig)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"An error occurred during the prediction process: {str(e)}")

# 侧边栏添加说明
with st.sidebar:
    st.markdown("""
    **Instructions:**
    1. Input the concentration values of five amino acids
    2. Click the "Predict Survival Rate" button
    3. View forecast results and survival curves

    **Reference range of biomarkers:**
    - PGA: 0.5-15 μmol/L
    - PRO: 1-20 μmol/L
    - ALA: 2-25 μmol/L
    - GLY: 5-30 μmol/L
    - HIS: 3-18 μmol/L
    """)

    st.markdown("---")
    st.markdown("**Regarding the DeepHit model**")
    st.markdown("""
    This is a deep learning based survival analysis model used to predict 
    the changes in survival probability of sepsis patients over time.
    """)

# 添加页脚
st.markdown("---")
st.caption("© 2025 DeepHit Survival Prediction System - For Medical Professionals Only")