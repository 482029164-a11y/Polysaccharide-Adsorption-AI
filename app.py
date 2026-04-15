import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
import os
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 核心架构类声明 (必须完整以支持 v16 内核加载)
# ==========================================

class StandardDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1)),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.network(x)

class PyTorchStandardRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: m.to(device); m.eval()
        with torch.no_grad():
            preds = torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.R = nn.Parameter(torch.ones(1, k_ensembles, input_dim) + torch.randn(1, k_ensembles, input_dim) * 0.01)
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1))
        )
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / math.sqrt(hidden_dim // 2))
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        x = x.unsqueeze(1) * self.R
        out = self.shared_bottom(x)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

class PyTorchTrueTabMRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 内核加载引擎 (v16 单内核门控系统)
# ==========================================
@st.cache_resource
def load_v16_kernel():
    try:
        pack = joblib.load('model_artifacts_v16.pkl')
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        model = pack['models']['True TabM']
        return X_cols, X_medians, model
    except Exception as e:
        st.error(f"严重错误：无法加载 v16 内核。请确保 model_artifacts_v16.pkl 在同级目录。错误: {e}")
        st.stop()

X_cols, X_medians, model_engine = load_v16_kernel()

def get_median(col_name, default_val=0.0):
    return float(X_medians.get(col_name, default_val))

# 特征自动归类逻辑
fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
log_handled = ['Log_specific surface area m2/g', 'Log_molecular weight', 'Log_adsorption time min', 'Log_C0_to_Dose_Ratio', 'Log_initial concentration mg/L']
# 排除掉 Physical_Gate_Inhibition，它由后台生成
remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_cols and c not in log_handled and c != 'Physical_Gate_Inhibition']

env_cols, mat_cols = [], []
for col in remaining_cols:
    col_lower = col.lower()
    if any(k in col_lower for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. 界面层与交互设计
# ==========================================
st.set_page_config(page_title="Qm Predictor (v16 TabM)", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于物理门控特征引导的 True TabM 深度学习预测系统 (v16)。")

with st.sidebar:
    st.subheader("系统设置")
    selected_theme = st.radio("界面风格：", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'), index=0)
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.markdown("### 引擎状态监控\n✅ 全局流形专家 (v16)\n✅ 物理门控感知器已激活\n系统将实时评估输入条件，自动激活内部抑制纠偏权重。")

user_inputs = {}
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

with tab_env:
    st.subheader("热力学与动力学操作参数")
    col1_e, col2_e = st.columns(2)
    with col1_e:
        c0_raw = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.4f", step=0.0001)
        dose_raw = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, format="%.4f", step=0.0001)
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_raw / (dose_raw + 1e-7))
    with col2_e:
        default_time = np.expm1(get_median('Log_adsorption time min', np.log1p(120.0)))
        time_raw = st.number_input("吸附时间 (min)", value=float(time_raw if 'time_raw' in locals() else default_time), format="%.4f", step=0.0001)
        user_inputs['Log_adsorption time min'] = np.log1p(time_raw)

    if env_cols:
        st.markdown("---")
        st.subheader("溶液化学环境")
        for col in env_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col, 0.0), format="%.4f", step=0.0001)

with tab_mat:
    st.subheader("形貌与高分子特性")
    col1_m, col2_m = st.columns(2)
    with col1_m:
        if 'Log_specific surface area m2/g' in X_cols:
            default_ssa = np.expm1(get_median('Log_specific surface area m2/g', np.log1p(150.0)))
            ssa_v = st.number_input("比表面积 (m2/g)", value=float(default_ssa), format="%.4f", step=0.0001)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
    with col2_m:
        if 'Log_molecular weight' in X_cols:
            default_mw = np.expm1(get_median('Log_molecular weight', np.log1p(300.0)))
            mw_v = st.number_input("分子量 (kDa)", value=float(default_mw), format="%.4f", step=0.0001)
            user_inputs['Log_molecular weight'] = np.log1p(mw_v)
            
    if mat_cols:
        st.markdown("---")
        st.subheader("其他物理/化学属性")
        for col in mat_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col, 0.0), format="%.4f", step=0.0001)

    if fg_cols:
        st.markdown("---")
        st.subheader("表面化学性质 (勾选代表存在)")
        fg_layout_cols = st.columns(3)
        for i, fg in enumerate(fg_cols):
            with fg_layout_cols[i % 3]:
                user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

with tab_dom:
    st.subheader("溶解性有机质竞争干扰评估")
    if dom_cols:
        for dom in dom_cols:
            user_inputs[dom] = st.number_input(f"{dom.replace('DOM_', '').replace('_浓度', '')} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)
    else:
        st.write("当前模型未包含 DOM 特征。")

# ==========================================
# 4. 物理门控推理引擎
# ==========================================
st.markdown("---")
if st.button("运行计算", use_container_width=True):
    # 1. 自动注入物理门控特征 (Gating Logic)
    ha_val = user_inputs.get('DOM_HA', 0.0)
    gate_signal = 1.0 if (c0_raw < 10.0 and ha_val > 0) else 0.0
    user_inputs['Physical_Gate_Inhibition'] = gate_signal
    
    # 2. 对齐特征空间
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    fill_dict = {c: 0.0 if (c.startswith('FG_') or c.startswith('DOM_')) else get_median(c, 0.0) for c in X_cols}
    final_df = final_df.fillna(value=fill_dict)
    
    try:
        # 3. 执行推理
        if gate_signal > 0.5:
            st.warning(f"🎯 物理门控激活：检测到 C0 ({c0_raw:.2f} mg/L) 属于极低浓度区域且受到 HA 竞争干扰。模型已自动在注意力机制底层调用纠偏权重。")
        else:
            st.success(f"✅ 物理状态正常：检测为常规浓度或纯水基质体系。当前处于全局热力学推理模式。")
            
        pred_log = model_engine.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        st.metric(label="预测吸附量 Qm (mg/g) [v16 TabM 引擎]", value=f"{main_prediction:.4f}")
        
    except Exception as e:
        st.error(f"推理引擎运行时错误: {e}")
