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
# 1. 核心架构类声明 (必须兼容 v6.2 与 v17 双内核)
# ==========================================

# --- 基础组件 ---
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

# --- v6.2 常规热力学流形核心 ---
class TrueTabMMini(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=128, k_ensembles=32, dropout=0.1):
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

# --- v17 极限纠偏流形核心 (Gated 架构) ---
class GatedTrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2), nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim), nn.Sigmoid()
        )
        self.R = nn.Parameter(torch.ones(1, k_ensembles, input_dim) + torch.randn(1, k_ensembles, input_dim) * 0.01)
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1))
        )
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / math.sqrt(hidden_dim // 2))
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        gate = self.feature_gate(x)
        x_gated = x * gate
        x_expanded = x_gated.unsqueeze(1) * self.R
        out = self.shared_bottom(x_expanded)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

class PyTorchGatedTabM(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# --- 兼容其他模型的占位与预测定义 (防 joblib 报错) ---
class PyTorchSingleDNN(BaseEstimator, RegressorMixin):
    def predict(self, X): pass
class PyTorchDeepEnsemble(BaseEstimator, RegressorMixin):
    def predict(self, X): pass
class PyTorchStandardRegressor(BaseEstimator, RegressorMixin):
    def predict(self, X): pass
class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    def predict(self, X): pass

# 🚀 注入 __main__ 命名空间
import __main__
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.GatedTrueTabMMini = GatedTrueTabMMini
__main__.PyTorchGatedTabM = PyTorchGatedTabM
__main__.PyTorchSingleDNN = PyTorchSingleDNN
__main__.PyTorchDeepEnsemble = PyTorchDeepEnsemble
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor

# ==========================================
# 2. 双路专家加载引擎 (v6.2 & v17)
# ==========================================
@st.cache_resource
def load_dual_expert_system():
    try:
        # 路径 1: 常规流形 (v6.2)
        pack_normal = joblib.load('model_artifacts_v6_2.pkl')
        # 路径 2: 特征门控纠偏流形 (v17)
        pack_penalty = joblib.load('model_artifacts_v17.pkl')
        
        tabm_normal = pack_normal['models']['True TabM']
        tabm_penalty = pack_penalty['models']['True TabM (Gated v17)']
        
        X_cols_v17 = pack_penalty['X'].columns.tolist()
        X_cols_v62 = pack_normal['X'].columns.tolist()
        X_medians = pack_penalty['X'].median(numeric_only=True).to_dict()
        
        return X_cols_v17, X_cols_v62, X_medians, tabm_normal, tabm_penalty
    except Exception as e:
        st.error(f"内核加载失败。请确保同级目录下有 v6_2 和 v17 的 pkl 文件。详情: {e}")
        st.stop()

X_cols_v17, X_cols_v62, X_medians, model_normal, model_penalty = load_dual_expert_system()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

# 动态特征归类 (以 v17 最全特征集为基准，排除后台控制列)
fg_cols = [c for c in X_cols_v17 if c.startswith('FG_')]
dom_cols = [c for c in X_cols_v17 if c.startswith('DOM_')]
log_handled = ['Log_specific surface area m2/g', 'Log_molecular weight', 'Log_adsorption time min', 'Log_C0_to_Dose_Ratio']
remaining_cols = [c for c in X_cols_v17 if c not in fg_cols and c not in dom_cols and c not in log_handled and c != 'Physical_Gate_Inhibition']

env_cols, mat_cols = [], []
for col in remaining_cols:
    col_lower = col.lower()
    if any(k in col_lower for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. 界面排版锁定
# ==========================================
st.set_page_config(page_title="Qm Predictor (Dual-MoE v17)", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于物理先验引导的混合专家模型 (Hard-Routing MoE)，集成 **Gated-TabM (v17)** 作为极值纠偏引擎。")

with st.sidebar:
    st.subheader("系统控制")
    selected_theme = st.radio("界面风格：", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'), index=0)
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.markdown("### 引擎调度监控\n✅ 常规专家: True TabM (v6.2)\n✅ 纠偏专家: Gated TabM (v17)\n物理门控将根据 C0 与 HA 浓度自动执行路由。")

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
        default_time = np.expm1(get_median('Log_adsorption time min'))
        time_raw = st.number_input("吸附时间 (min)", value=float(default_time if default_time > 0 else 120.0), format="%.4f", step=0.0001)
        user_inputs['Log_adsorption time min'] = np.log1p(time_raw)

    if env_cols:
        st.markdown("---")
        st.subheader("溶液化学环境")
        for col in env_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

with tab_mat:
    st.subheader("形貌与高分子特性")
    col1_m, col2_m = st.columns(2)
    with col1_m:
        if 'Log_specific surface area m2/g' in X_cols_v17:
            ssa_def = np.expm1(get_median('Log_specific surface area m2/g'))
            ssa_v = st.number_input("比表面积 (m2/g)", value=float(ssa_def if ssa_def > 0 else 150.0), format="%.4f", step=0.0001)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
    with col2_m:
        if 'Log_molecular weight' in X_cols_v17:
            mw_def = np.expm1(get_median('Log_molecular weight'))
            mw_v = st.number_input("分子量 (kDa)", value=float(mw_def if mw_def > 0 else 300.0), format="%.4f", step=0.0001)
            user_inputs['Log_molecular weight'] = np.log1p(mw_v)
            
    if mat_cols:
        st.markdown("---")
        st.subheader("其他物理/化学属性")
        for col in mat_cols:
            user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

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

# ==========================================
# 4. 双路硬路由推理核心
# ==========================================
st.markdown("---")
if st.button("运行预测引擎", use_container_width=True):
    ha_val = user_inputs.get('DOM_HA', 0.0)
    
    try:
        # 核心物理边界判定
        if c0_raw < 10.0 and ha_val > 0.0:
            st.warning(f"⚠️ 物理门控触发：检测到 C0 ({c0_raw:.2f} mg/L) 属于极低浓度区域且伴随 HA 竞争干扰。底层调度已将张量流重定向至【极值纠偏专家 Gated-TabM (v17)】，特征注意力门控已开启。")
            
            # v17 推理要求显式写入物理抑制特征
            user_inputs['Physical_Gate_Inhibition'] = 1.0
            final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols_v17)
            # 安全填补
            for col in X_cols_v17:
                if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
            
            pred_log = model_penalty.predict(final_df)[0]
            engine_used = "v17 Gated 流形 (特征关注强化)"
        else:
            st.success(f"✅ 物理门控就绪：检测为常规热力学空间。底层调度已将张量流重定向至【全局连续专家 True TabM (v6.2)】。")
            
            # v6.2 推理：严格对齐老版本的特征空间维度
            final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols_v62)
            for col in X_cols_v62:
                if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
                
            pred_log = model_normal.predict(final_df)[0]
            engine_used = "v6.2 全局流形"
            
        main_prediction = np.expm1(pred_log)
        st.metric(label=f"理论平衡吸附量 Qm (mg/g) [调度内核：{engine_used}]", value=f"{main_prediction:.4f}")
        
    except Exception as e:
        st.error(f"算力引擎调度失败，原因: {e}")
