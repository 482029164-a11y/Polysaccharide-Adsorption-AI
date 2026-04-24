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
# 1. 核心架构类声明 (V32 纯净回归版)
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
# 2. 模型加载 (V32 热力学锚点版)
# ==========================================
@st.cache_resource
def load_v32_system():
    try:
        pack = joblib.load('model_artifacts_final.pkl')
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        return X_cols, X_medians, pack['models']
    except Exception as e:
        st.error(f"严重错误：找不到内核文件 model_artifacts_v32.pkl。详细信息: {e}")
        st.stop()

X_cols, X_medians, models_dict = load_v32_system()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

# ==========================================
# 🚀 UI 去重与特征智能拦截机制
# ==========================================
# 后台全自动计算对数与比值，前端界面隐去这些组合变量
hidden_auto_calc_cols = [
    'initial concentration mg/L', 
    'Log_initial concentration mg/L',
    'adsorbent dose mg/ml', 
    'Log_adsorbent dose mg/ml',
    'C0_to_Dose_Ratio',
    'Log_C0_to_Dose_Ratio',
    'adsorption time min',
    'Log_adsorption time min',
    'specific surface area m2/g',
    'Log_specific surface area m2/g',
    'molecular weight',
    'Log_molecular weight'
]

fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_ratio_cols = [c for c in X_cols if '_to_C0_Ratio' in c]

# 提取真正的独立输入特征
remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_ratio_cols and c not in hidden_auto_calc_cols and not c.startswith('DOM_')]

env_cols, mat_cols = [], []
for col in remaining_cols:
    if any(k in col.lower() for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. UI 界面布局
# ==========================================
st.set_page_config(page_title="Qm Predictor (V32 旗舰版)", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于 **化学先验特征工程 (Stoichiometric Ratio)** 与 **代价敏感热力学锚点 (Cost-Sensitive Anchor)** 双重驱动的机器学习平台。")

with st.sidebar:
    st.subheader("系统控制")
    selected_theme = st.radio("界面风格：", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'), index=0)
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.subheader("🧠 推断引擎选择")
    selected_model_name = st.selectbox("请选择底层机器学习算法:", list(models_dict.keys()), index=4)
    st.markdown("---")
    st.markdown("✅ UI智能降噪脱水\n✅ V32 热力学锚点内核就绪\n✅ 前端物理惩罚逻辑已完全剥离")

user_inputs = {}
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

with tab_env:
    col1_e, col2_e = st.columns(2)
    with col1_e:
        c0_raw = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.4f", step=0.0001)
        dose_raw = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, format="%.4f", step=0.0001)
        
        # 后台自动派生全量 C0/Dose 特征
        user_inputs['initial concentration mg/L'] = c0_raw
        user_inputs['Log_initial concentration mg/L'] = np.log1p(c0_raw)
        user_inputs['adsorbent dose mg/ml'] = dose_raw
        user_inputs['Log_adsorbent dose mg/ml'] = np.log1p(dose_raw)
        user_inputs['C0_to_Dose_Ratio'] = c0_raw / (dose_raw + 1e-7)
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_raw / (dose_raw + 1e-7))
        
    with col2_e:
        default_time = np.expm1(get_median('Log_adsorption time min')) if get_median('Log_adsorption time min') > 0 else get_median('adsorption time min')
        time_raw = st.number_input("吸附时间 (min)", value=float(default_time if default_time > 0 else 120.0), format="%.4f", step=0.0001)
        user_inputs['adsorption time min'] = time_raw
        user_inputs['Log_adsorption time min'] = np.log1p(time_raw)

    if env_cols:
        st.markdown("---")
        for col in env_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

with tab_mat:
    col1_m, col2_m = st.columns(2)
    with col1_m:
        ssa_def = np.expm1(get_median('Log_specific surface area m2/g')) if get_median('Log_specific surface area m2/g') > 0 else get_median('specific surface area m2/g')
        ssa_v = st.number_input("比表面积 (m2/g)", value=float(ssa_def if ssa_def > 0 else 150.0), format="%.4f", step=0.0001)
        user_inputs['specific surface area m2/g'] = ssa_v
        user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
        
    with col2_m:
        mw_def = np.expm1(get_median('Log_molecular weight')) if get_median('Log_molecular weight') > 0 else get_median('molecular weight')
        mw_v = st.number_input("分子量 (kDa)", value=float(mw_def if mw_def > 0 else 300.0), format="%.4f", step=0.0001)
        user_inputs['molecular weight'] = mw_v
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
    # 动态捕获模型期望的 DOM 种类
    expected_doms = [c.replace('Log_', '').replace('_to_C0_Ratio', '') for c in dom_ratio_cols]
    
    if expected_doms:
        for dom in expected_doms:
            dom_raw_inputs[dom] = st.number_input(f"{dom} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)

# ==========================================
# 4. 纯净推断执行
# ==========================================
st.markdown("---")
if st.button(f"运行 [{selected_model_name}]", use_container_width=True):
    try:
        # 在后台动态计算多重 DOM / C0 配位比值
        for dom, dom_conc in dom_raw_inputs.items():
            user_inputs[f'DOM_{dom}'] = dom_conc
            user_inputs[f'Log_{dom}_to_C0_Ratio'] = np.log1p(dom_conc / (c0_raw + 1e-6))

        # 结构化张量并补齐未命中值
        final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
        for col in X_cols:
            if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
        
        # 加载并预测
        active_model = models_dict[selected_model_name]
        pred_log = active_model.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        
        st.success("✅ 推断完成！")
        st.metric(label=f"预测理论平衡吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        
        # 内部状态透视
        with st.expander("🔍 查看后台解析状态 (化学先验引擎)"):
            st.write(f"**初始输入浓度 C0:** `{c0_raw:.4f} mg/L`")
            for dom in expected_doms:
                ratio_val = user_inputs[f'Log_{dom}_to_C0_Ratio']
                st.write(f"**{dom}/C0 对数配位比:** `{ratio_val:.4f}`")
            st.caption("注：模型通过训练时的热力学锚点加权（Cost-Sensitive Anchors），已自主学得高配位比下的严重物理坍缩效应，前端无需任何硬编码补偿。")
            
    except Exception as e:
        st.error(f"引擎调度失败: {e}")
