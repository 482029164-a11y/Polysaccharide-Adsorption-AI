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
# 1. 核心架构类声明 (Scikit-Learn 严格兼容版)
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
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max
        self.T_0 = T_0; self.T_mult = T_mult

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
    def __init__(self, k_ensembles=5, epochs=200, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=40, T_mult=1.5):
        self.k_ensembles = k_ensembles; self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max; self.T_0 = T_0; self.T_mult = T_mult

    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: m.to(device); m.eval()
        with torch.no_grad():
            preds = torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=256, k_ensembles=32, dropout=0.1):
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
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max
        self.T_0 = T_0; self.T_mult = T_mult

    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# 🚀 注入 __main__ 空间，这是 joblib 反序列化的绝对前提
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 内核加载引擎
# ==========================================
@st.cache_resource
def load_v16_kernel():
    path = 'model_artifacts_v16.pkl'
    if not os.path.exists(path):
        st.error(f"找不到内核文件: {path}")
        st.stop()
    try:
        pack = joblib.load(path)
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        model = pack['models']['True TabM']
        return X_cols, X_medians, model
    except Exception as e:
        st.error(f"内核加载失败: {e}")
        st.stop()

X_cols, X_medians, model_engine = load_v16_kernel()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

# ==========================================
# 3. 界面排版锁定
# ==========================================
st.set_page_config(page_title="Qm Predictor v16", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统 (v16)")
st.markdown("基于物理门控特征引导的 True TabM 深度学习系统。")

with st.sidebar:
    st.subheader("系统控制")
    selected_theme = st.radio("界面风格", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'))
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.markdown("### 监控状态\n✅ 专家引擎: True TabM\n✅ 物理门控: 自动识别")

user_inputs = {}
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

# --- TAB 1: 环境 ---
with tab_env:
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        c0_v = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.2f")
        dose_v = st.number_input("投加量 Dose (mg/ml)", value=1.0, format="%.2f")
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_v / (dose_v + 1e-7))
    with col_e2:
        time_v = st.number_input("吸附时间 (min)", value=120.0, format="%.2f")
        user_inputs['Log_adsorption time min'] = np.log1p(time_v)
    
    st.markdown("---")
    for col in X_cols:
        col_lower = col.lower()
        # 排除掉材料、官能团、DOM和已处理列，剩下的放环境
        if any(k in col_lower for k in ['ph', 'temp', 'rpm', 'speed']) and col not in user_inputs:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f")

# --- TAB 2: 材料 (锁定 FG_Phosphate) ---
with tab_mat:
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if 'Log_specific surface area m2/g' in X_cols:
            ssa_v = st.number_input("比表面积 (m2/g)", value=float(np.expm1(get_median('Log_specific surface area m2/g'))))
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
    with col_m2:
        if 'Log_molecular weight' in X_cols:
            mw_v = st.number_input("分子量 (kDa)", value=float(np.expm1(get_median('Log_molecular weight'))))
            user_inputs['Log_molecular weight'] = np.log1p(mw_v)
    
    st.markdown("---")
    st.subheader("表面官能团 (FG)")
    fg_cols = [c for c in X_cols if c.startswith('FG_')]
    if fg_cols:
        fg_layout = st.columns(3)
        for i, fg in enumerate(fg_cols):
            with fg_layout[i % 3]:
                user_inputs[fg] = float(st.checkbox(fg.replace('FG_', ''), key=fg))
    
    st.markdown("---")
    st.subheader("元素组成与其他物理指标")
    mat_keys = ['carbon', 'oxygen', 'nitrogen', 'ash', 'pore', 'porosity']
    for col in X_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in mat_keys) and col not in user_inputs and not col.startswith('FG_'):
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f")

# --- TAB 3: DOM ---
with tab_dom:
    dom_cols = [c for c in X_cols if c.startswith('DOM_')]
    for col in dom_cols:
        user_inputs[col] = st.number_input(f"{col.replace('DOM_', '')} 浓度 (mg/L)", value=0.0, format="%.4f")

# ==========================================
# 4. 推理核心
# ==========================================
st.markdown("---")
if st.button("开始运行 v16 推理", use_container_width=True):
    # 自动生成门控信号
    ha_val = user_inputs.get('DOM_HA', 0.0)
    gate_signal = 1.0 if (c0_v < 10.0 and ha_val > 0) else 0.0
    user_inputs['Physical_Gate_Inhibition'] = gate_signal
    
    # 构建并对齐
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    for col in X_cols:
        if pd.isna(final_df[col][0]):
            final_df[col] = get_median(col)
            
    try:
        if gate_signal > 0.5:
            st.warning("🎯 物理门控激活：极低浓度竞争体系抑制修正中...")
        else:
            st.success("✅ 全局流形模式运行正常")
            
        # 此处 model_engine 是一个 Pipeline，它会自动调用 PyTorchTrueTabMRegressor.predict
        pred_log = model_engine.predict(final_df)[0]
        st.metric(label="预测吸附量 Qm (mg/g)", value=f"{np.expm1(pred_log):.4f}")
    except Exception as e:
        st.error(f"推理引擎异常: {e}")
