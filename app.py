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
# 1. 核心架构类声明 (V31 纯净版，严密对齐训练内核)
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

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, k_ensembles=32, dropout=0.1):
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
    def fit(self, X, y=None, **kwargs): return self
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        with torch.no_grad(): return self.model_(X_t).mean(dim=1).cpu().numpy().flatten()

class PyTorchSingleDNN(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def fit(self, X, y=None, **kwargs): return self
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        with torch.no_grad(): return self.model_(X_t).cpu().numpy().flatten()

class PyTorchDeepEnsemble(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def fit(self, X, y=None, **kwargs): return self
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        for m in self.models_: m.to(device); m.eval()
        with torch.no_grad():
            return torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()

import __main__
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchSingleDNN = PyTorchSingleDNN
__main__.PyTorchDeepEnsemble = PyTorchDeepEnsemble

# ==========================================
# 2. 模型加载 (V31 纯净先验版)
# ==========================================
@st.cache_resource
def load_v31_system():
    try:
        pack = joblib.load('model_artifacts_v31.pkl')
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        return X_cols, X_medians, pack['models']
    except Exception as e:
        st.error(f"严重错误：找不到内核文件 model_artifacts_v31.pkl。详细信息: {e}")
        st.stop()

X_cols, X_medians, models_dict = load_v31_system()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

# 分离特征展示
fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_ratio_cols = [c for c in X_cols if '_to_C0_Ratio' in c]
log_handled = ['Log_specific surface area m2/g', 'Log_molecular weight', 'Log_adsorption time min', 'Log_C0_to_Dose_Ratio']
remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_ratio_cols and c not in log_handled and not c.startswith('DOM_')]

env_cols, mat_cols = [], []
for col in remaining_cols:
    if any(k in col.lower() for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. UI 界面布局
# ==========================================
st.set_page_config(page_title="Qm Predictor (V31 化学先验版)", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于领域先验引导的机器学习平台。内核集成 **DOM/重金属化学计量比 (Stoichiometric Ratio)** 显式映射，无需硬编码即可自动感知极端热力学突变。")

with st.sidebar:
    st.subheader("系统控制")
    selected_theme = st.radio("界面风格：", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'), index=0)
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.subheader("🧠 推断引擎选择")
    # 让用户可以随时切换五大模型对比
    selected_model_name = st.selectbox("请选择底层机器学习算法:", list(models_dict.keys()), index=4)
    st.markdown("---")
    st.markdown("✅ 数据管道纯净化\n✅ 物理门控已下线\n✅ 化学先验特征交叉已激活")

user_inputs = {}
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

with tab_env:
    col1_e, col2_e = st.columns(2)
    with col1_e:
        c0_raw = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.4f", step=0.0001)
        dose_raw = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, format="%.4f", step=0.0001)
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_raw / (dose_raw + 1e-7))
    with col2_e:
        default_time = np.expm1(get_median('Log_adsorption time min'))
        time_raw = st.number_input("吸附时间 (min)", value=float(default_time if default_time > 0 else 120.0), format="%.4f", step=0.0001)
        user_inputs['Log_adsorption time min'] = np.log1p(time_raw)

    if env_cols:
        st.markdown("---")
        for col in env_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

with tab_mat:
    col1_m, col2_m = st.columns(2)
    with col1_m:
        if 'Log_specific surface area m2/g' in X_cols:
            ssa_def = np.expm1(get_median('Log_specific surface area m2/g'))
            ssa_v = st.number_input("比表面积 (m2/g)", value=float(ssa_def if ssa_def > 0 else 150.0), format="%.4f", step=0.0001)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
    with col2_m:
        if 'Log_molecular weight' in X_cols:
            mw_def = np.expm1(get_median('Log_molecular weight'))
            mw_v = st.number_input("分子量 (kDa)", value=float(mw_def if mw_def > 0 else 300.0), format="%.4f", step=0.0001)
            user_inputs['Log_molecular weight'] = np.log1p(mw_v)
            
    if mat_cols:
        st.markdown("---")
        for col in mat_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

    if fg_cols:
        st.markdown("---")
        fg_layout_cols = st.columns(3)
        for i, fg in enumerate(fg_cols):
            with fg_layout_cols[i % 3]:
                user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

with tab_dom:
    dom_raw_inputs = {}
    # 动态捕获模型需要的 DOM 种类 (支持 HA, FA, CA 等)
    expected_doms = [c.replace('Log_', '').replace('_to_C0_Ratio', '') for c in dom_ratio_cols]
    
    if expected_doms:
        for dom in expected_doms:
            dom_raw_inputs[dom] = st.number_input(f"{dom} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)

# ==========================================
# 4. 纯净化学推断执行
# ==========================================
st.markdown("---")
if st.button(f"运行 [{selected_model_name}]", use_container_width=True):
    try:
        # 🔥 V31 核心逻辑：在前端仅作无量纲化学交叉，绝不干涉模型权重
        for dom, dom_conc in dom_raw_inputs.items():
            # 恢复训练时的数据流对齐
            user_inputs[f'DOM_{dom}'] = dom_conc
            user_inputs[f'Log_{dom}_to_C0_Ratio'] = np.log1p(dom_conc / (c0_raw + 1e-6))

        # 整理张量
        final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
        for col in X_cols:
            if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
        
        # 提取选定的模型
        active_model = models_dict[selected_model_name]
        
        # 纯净推断
        pred_log = active_model.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        
        st.success("✅ 推断完成！")
        st.metric(label=f"预测理论平衡吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        
        # 调试/展示信息
        with st.expander("🔍 查看化学先验特征解析状态"):
            for dom in expected_doms:
                ratio_val = user_inputs[f'Log_{dom}_to_C0_Ratio']
                st.write(f"**{dom}/C0 对数计量比 (Log_Ratio):** `{ratio_val:.4f}`")
            st.caption("注：模型完全依赖上述化学先验比例自主判定系统处于架桥主导（低比值突变）还是位阻主导（高比值坍缩）状态，已剥离所有人工硬编码干预。")
            
    except Exception as e:
        st.error(f"引擎调度失败: {e}")
