import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

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

@st.cache_resource
def load_v33_system():
    try:
        pack = joblib.load('model_artifacts_v33.pkl')
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        return X_cols, X_medians, pack['models']
    except Exception as e:
        st.error(f"严重错误：找不到内核文件 model_artifacts_v33.pkl。详细信息: {e}")
        st.stop()

X_cols, X_medians, models_dict = load_v33_system()

def get_median(col_name):
    return float(X_medians.get(col_name, 0.0))

hidden_auto_calc_cols = [
    'initial concentration mg/L', 'Log_initial concentration mg/L',
    'adsorbent dose mg/ml', 'Log_adsorbent dose mg/ml',
    'C0_to_Dose_Ratio', 'Log_C0_to_Dose_Ratio',
    'adsorption time min', 'Log_adsorption time min',
    'specific surface area m2/g', 'Log_specific surface area m2/g',
    'molecular weight', 'Log_molecular weight'
]

fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_ratio_cols = [c for c in X_cols if '_to_C0_Ratio' in c and 'per_SSA' not in c]

remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_ratio_cols and c not in hidden_auto_calc_cols and not c.startswith('DOM_') and 'per_SSA' not in c]

env_cols, mat_cols = [], []
for col in remaining_cols:
    if any(k in col.lower() for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

st.set_page_config(page_title="Qm Predictor (V33 旗舰版)", layout="wide")

st.title("目标污染物吸附性能预测系统")
st.markdown("基于 **SSA归一化化学计量比 (Surface-Area Normalized Stoichiometric Ratio)** 驱动的机器学习平台。")

with st.sidebar:
    st.subheader("🧠 推断引擎选择")
    selected_model_name = st.selectbox("请选择底层机器学习算法:", list(models_dict.keys()), index=4)

user_inputs = {}
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

with tab_env:
    col1_e, col2_e = st.columns(2)
    with col1_e:
        c0_raw = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.4f", step=0.0001)
        dose_raw = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, format="%.4f", step=0.0001)
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

    for col in env_cols:
        user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

with tab_mat:
    col1_m, col2_m = st.columns(2)
    with col1_m:
        ssa_def = np.expm1(get_median('Log_specific surface area m2/g')) if get_median('Log_specific surface area m2/g') > 0 else get_median('specific surface area m2/g')
        ssa_raw = st.number_input("比表面积 (m2/g)", value=float(ssa_def if ssa_def > 0 else 150.0), format="%.4f", step=0.0001)
        user_inputs['specific surface area m2/g'] = ssa_raw
        user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_raw)
        
    with col2_m:
        mw_def = np.expm1(get_median('Log_molecular weight')) if get_median('Log_molecular weight') > 0 else get_median('molecular weight')
        mw_v = st.number_input("分子量 (kDa)", value=float(mw_def if mw_def > 0 else 300.0), format="%.4f", step=0.0001)
        user_inputs['molecular weight'] = mw_v
        user_inputs['Log_molecular weight'] = np.log1p(mw_v)
            
    for col in mat_cols:
        user_inputs[col] = st.number_input(col, value=get_median(col), format="%.4f", step=0.0001)

    fg_layout_cols = st.columns(3)
    for i, fg in enumerate(fg_cols):
        with fg_layout_cols[i % 3]:
            user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

with tab_dom:
    dom_raw_inputs = {}
    expected_doms = [c.replace('Log_', '').replace('_to_C0_Ratio', '') for c in dom_ratio_cols]
    for dom in expected_doms:
        dom_raw_inputs[dom] = st.number_input(f"{dom} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)

st.markdown("---")
if st.button(f"运行 [{selected_model_name}]", use_container_width=True):
    try:
        for dom, dom_conc in dom_raw_inputs.items():
            user_inputs[f'DOM_{dom}'] = dom_conc
            ratio_val = dom_conc / (c0_raw + 1e-6)
            user_inputs[f'Log_{dom}_to_C0_Ratio'] = np.log1p(ratio_val)
            user_inputs[f'Log_{dom}_to_C0_Ratio_per_SSA'] = np.log1p(ratio_val / (ssa_raw + 1e-5))

        final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
        for col in X_cols:
            if pd.isna(final_df[col][0]): final_df[col] = get_median(col)
        
        active_model = models_dict[selected_model_name]
        pred_log = active_model.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        
        st.success("✅ 推断完成！")
        st.metric(label=f"预测理论平衡吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        
        with st.expander("🔍 查看归一化解析状态"):
            for dom in expected_doms:
                st.write(f"**{dom}/C0 配位比:** `{user_inputs[f'Log_{dom}_to_C0_Ratio']:.4f}`")
                st.write(f"**单位比表面积配位负载 (per_SSA):** `{user_inputs[f'Log_{dom}_to_C0_Ratio_per_SSA']:.6f}`")
            
    except Exception as e:
        st.error(f"引擎调度失败: {e}")
