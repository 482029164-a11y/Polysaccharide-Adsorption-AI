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
# 1. 核心架构类声明 (v27 Symbolic PINN 严格兼容版)
# ==========================================

# --- 基础网络组件 ---
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

# --- v6.2 常规热力学专家核心 ---
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
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size; self.lr_min = lr_min; self.lr_max = lr_max; self.T_0 = T_0; self.T_mult = T_mult
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# --- v27 显式符号物理方程纠偏专家核心 (Symbolic PINN) ---
class GatedTrueTabMMini(nn.Module):
    def __init__(self, input_dim, pro_idx, inh_idx, hidden_dim=256, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.pro_idx = pro_idx
        self.inh_idx = inh_idx
        
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
        
        # 🔥 v27: 显式物理阻力权重参数对齐
        self.promo_scale = nn.Parameter(torch.tensor([2.0]))
        self.inhib_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        gate = self.feature_gate(x)
        x_expanded = (x * gate).unsqueeze(1) * self.R
        out = self.shared_bottom(x_expanded)
        deep_out = (out * self.head_weights).sum(dim=-1) + self.head_biases
        
        # 精准定位张量内部的特征并执行显式强制计算
        pro_feature = x[:, self.pro_idx].unsqueeze(-1)
        inh_feature = x[:, self.inh_idx].unsqueeze(-1)
        
        promo_effect = torch.abs(self.promo_scale) * pro_feature
        inhib_effect = torch.abs(self.inhib_scale) * inh_feature
        
        # 强制加减方向
        final_out = deep_out + promo_effect - inhib_effect
        return final_out

class PyTorchGatedTabM(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=150, batch_size=32, lr=0.002):
        self.epochs = epochs; self.batch_size = batch_size; self.lr = lr
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_v = X.values if hasattr(X, 'values') else X
        X_t = torch.tensor(X_v, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# --- 其他防报错占位类 ---
class PyTorchSingleDNN(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=80, batch_size=32, lr=0.001):
        self.epochs = epochs; self.batch_size = batch_size; self.lr = lr
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X): pass

class PyTorchDeepEnsemble(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, k=3, epochs=80, batch_size=32, lr=0.001):
        self.k = k; self.epochs = epochs; self.batch_size = batch_size; self.lr = lr
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X): pass

class PyTorchStandardRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5): pass
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X): pass

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, k_ensembles=5, epochs=200, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=40, T_mult=1.5): pass
    def fit(self, X, y=None, **kwargs): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X): pass

# 🚀 全域挂载：确保 Joblib 读取时类定义完全对齐
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
# 2. 双路模型加载 (v6.2 & v27)
# ==========================================
@st.cache_resource
def load_dual_expert_system():
    try:
        pack_normal = joblib.load('model_artifacts_v6_2.pkl')
        pack_penalty = joblib.load('model_artifacts_v27.pkl')
        
        tabm_normal = pack_normal['models']['True TabM']
        # 对应 v27 保存时的键名
        tabm_penalty = pack_penalty['models']['True TabM (Symbolic PINN v27)']
        
        X_cols_v27 = pack_penalty['X'].columns.tolist()
        X_cols_v62 = pack_normal['X'].columns.tolist()
        X_medians = pack_penalty['X'].median(numeric_only=True).to_dict()
        
        return X_cols_v27, X_cols_v62, X_medians, tabm_normal, tabm_penalty
    except Exception as e:
        st.error(f"严重错误：找不到内核文件。确保 v6_2 和 v27 的 pkl 文件在同级目录下。详细信息: {e}")
        st.stop()

X_cols_v27, X_cols_v62, X_medians, model_normal, model_penalty = load_dual_expert_system()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

# 这些显式物理特征对于网络计算必不可少，但不需要展示给前端用户
hidden_cols = ['Physical_Gate_Inhibition', 'HA_Bridging_Promotion', 'HA_Competitive_Inhibition', 'Asymptotic_Inhibition_Force']
fg_cols = [c for c in X_cols_v27 if c.startswith('FG_')]
dom_cols = [c for c in X_cols_v27 if c.startswith('DOM_')]
log_handled = ['Log_specific surface area m2/g', 'Log_molecular weight', 'Log_adsorption time min', 'Log_C0_to_Dose_Ratio']
remaining_cols = [c for c in X_cols_v27 if c not in fg_cols and c not in dom_cols and c not in log_handled and c not in hidden_cols]

env_cols, mat_cols = [], []
for col in remaining_cols:
    if any(k in col.lower() for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. UI 界面布局
# ==========================================
st.set_page_config(page_title="Qm Predictor (v27 显式物理方程版)", layout="wide")

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        st.markdown("<style>.stApp { background-color: #1E1E1E; color: #FFFFFF; }</style>", unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        st.markdown("<style>.stApp { background-color: #FAEDDF; color: #4A3A2C; }</style>", unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于物理先验引导的混合专家模型 (Hard-Routing MoE)，集成 **v27 显式符号物理方程嵌入 (Symbolic PINN)** 的 Gated TabM 引擎。")

with st.sidebar:
    st.subheader("系统控制")
    selected_theme = st.radio("界面风格：", ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'), index=0)
    apply_custom_theme(selected_theme)
    st.markdown("---")
    st.markdown("### 引擎调度监控\n✅ 常规专家: True TabM (v6.2)\n✅ 显式方程专家: Gated TabM (v27)\n系统将自动推演双态物理力场，直接从网络末端硬性拉升或坍缩预测值。")

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
        if 'Log_specific surface area m2/g' in X_cols_v27:
            ssa_def = np.expm1(get_median('Log_specific surface area m2/g'))
            ssa_v = st.number_input("比表面积 (m2/g)", value=float(ssa_def if ssa_def > 0 else 150.0), format="%.4f", step=0.0001)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_v)
    with col2_m:
        if 'Log_molecular weight' in X_cols_v27:
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
    if dom_cols:
        for dom in dom_cols:
            user_inputs[dom] = st.number_input(f"{dom.replace('DOM_', '').replace('_浓度', '')} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)

# ==========================================
# 4. 双路硬路由与物理公式演算
# ==========================================
st.markdown("---")
if st.button("运行混合专家系统", use_container_width=True):
    ha_val = user_inputs.get('DOM_HA', 0.0)
    
    try:
        if c0_raw < 10.0 and ha_val > 0.0:
            # v27 核心公式演算与前置特征注入
            bridging_promo = np.exp(-((ha_val - 10.0)**2) / 5.0) if c0_raw < 10.0 else 0.0
            comp_inhib = max(0.0, ha_val - 10.0) * (1.0 - np.tanh(c0_raw / 10.0))
            
            st.warning(f"⚠️ 物理门控已激活：检测到极低浓度且存在 HA 竞争。")
            st.info(f"🧠 系统演算：提取架桥促进特征 {bridging_promo:.4f}，提取竞争抑制特征 {comp_inhib:.4f}。底层已切换至【v27 显式符号 PINN 专家】进行直接物理运算。")
            
            user_inputs['Physical_Gate_Inhibition'] = 1.0
            user_inputs['HA_Bridging_Promotion'] = bridging_promo
            user_inputs['HA_Competitive_Inhibition'] = comp_inhib
            
            final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols_v27)
            for col in X_cols_v27:
                if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
            
            # 模型执行 predict 时，标准化流程完毕后的特征张量将被送入 forward，
            # 内部会自动根据索引抓取这两个物理特征，应用 promo_scale 和 inhib_scale，执行显式方程
            pred_log = model_penalty.predict(final_df)[0]
            engine_used = "v27 Gated-TabM (Symbolic PINN 强制干预模式)"
            
        else:
            st.success(f"✅ 物理状态稳定：检测为常规浓度或纯水基质。底层切换至【v6.2 全局热力学专家】以保证流形的平滑预测。")
            
            final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols_v62)
            for col in X_cols_v62:
                if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
                
            pred_log = model_normal.predict(final_df)[0]
            engine_used = "v6.2 全局流形专家"
            
        main_prediction = np.expm1(pred_log)
        st.metric(label=f"预测理论平衡吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        st.caption(f"当前执行内核: {engine_used}")
        
    except Exception as e:
        st.error(f"引擎调度失败: {e}")
