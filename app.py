import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import sys
import os
import optuna
import xgboost
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 底层蓝图 (严密对齐内核)
# ==========================================
class StandardDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1)), nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.network(x)

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.R = nn.Parameter(torch.ones(1, k_ensembles, input_dim) + torch.randn(1, k_ensembles, input_dim) * 0.01)
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1))
        )
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / 11.31)
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        x = x.unsqueeze(1) * self.R
        out = self.shared_bottom(x)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

class PyTorchBaseWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, lr=0.001, batch_size=32):
        self.epochs = epochs; self.lr = lr; self.batch_size = batch_size
    def fit(self, X, y=None): return self

class PyTorchSingleDNN(PyTorchBaseWrapper):
    def predict(self, X):
        if hasattr(self, 'model_'): self.model_.to('cpu').eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        with torch.no_grad(): return self.model_(X_t).cpu().numpy().flatten()

class PyTorchDeepEnsemble(PyTorchBaseWrapper):
    def predict(self, X):
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        for m in self.models_: m.to('cpu').eval()
        with torch.no_grad():
            return torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()

class PyTorchTrueTabMRegressor(PyTorchBaseWrapper):
    def predict(self, X):
        if hasattr(self, 'model_'): self.model_.to('cpu').eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        with torch.no_grad(): return self.model_(X_t).mean(dim=1).cpu().numpy().flatten()

sys.modules['__main__'].StandardDNN = StandardDNN
sys.modules['__main__'].TrueTabMMini = TrueTabMMini
sys.modules['__main__'].PyTorchSingleDNN = PyTorchSingleDNN
sys.modules['__main__'].PyTorchDeepEnsemble = PyTorchDeepEnsemble
sys.modules['__main__'].PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor

# ==========================================
# 1. 安全内核加载
# ==========================================
@st.cache_resource
def load_models():
    # 彻底杜绝 map_location 多重赋值报错
    if not hasattr(torch, '_orig_load_backup'):
        torch._orig_load_backup = torch.load

    def safe_cpu_load(*args, **kwargs):
        kwargs.pop('map_location', None)
        if len(args) >= 2:
            args_list = list(args)
            args_list[1] = 'cpu'
            return torch._orig_load_backup(*tuple(args_list), **kwargs)
        else:
            kwargs['map_location'] = 'cpu'
            return torch._orig_load_backup(*args, **kwargs)

    torch.load = safe_cpu_load
    try:
        f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
        data = joblib.load(f_path)
    finally:
        torch.load = torch._orig_load_backup
    
    return data['X'], data['models']

try:
    X_base, models = load_models()
except Exception as e:
    st.error(f"内核读取失败: {e}")
    st.stop()

# ==========================================
# 2. 绝对精准：基于表头的特征写死清单
# ==========================================
# A. 需要用户输入的 14 个连续物理量
continuous_inputs = [
    'Polysaccharide content', 'surface charge mv', 'porosity', 'pore size um', 
    'pH', 'adsorbent dose mg/ml', 'temperature k', 'V ml', 
    'DOM_CA', 'DOM_FA', 'DOM_HA', 
    'adsorption time min', 'specific surface area m2/g', 'initial concentration mg/L'
]

# B. 需要用户勾选的 9 个官能团 (0/1)
functional_inputs = [
    'FG_Amide', 'FG_Amino', 'FG_Carbonyl', 'FG_Carboxyl', 'FG_Ether', 
    'FG_Hydroxyl', 'FG_Phosphate', 'FG_Siloxane', 'FG_Sulfonic acid'
]

# 动态反求训练集中位数作为默认值，确保输入框永远不会报错
def get_default_val(col_name):
    if col_name in X_base.columns:
        return float(X_base[col_name].median())
    # 如果原列不在，说明它被转成了 Log_ 形态，在此反向解构
    log_name = f"Log_{col_name}"
    if log_name in X_base.columns:
        return float(np.expm1(X_base[log_name].median()))
    return 0.0

# ==========================================
# 3. 交互界面设计 (超高速实时响应)
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

selected_name = st.selectbox("1. 选择预测中枢:", list(models.keys()))
st.divider()

st.subheader("2. 基础物理工况录入")
st.info("💡 参数修改后，结果将实现毫秒级实时刷新。派生比值已隐式接通。")

user_vals = {}

cols_p = st.columns(2)
for i, name in enumerate(continuous_inputs):
    with cols_p[i % 2]:
        def_val = get_default_val(name)
        user_vals[name] = st.number_input(f"{name}", value=def_val, format="%.4f")

st.markdown("---")
st.subheader("3. 表面官能团检测")
st.caption("勾选代表存在 (1)，不勾选代表未检出 (0)。")

cols_f = st.columns(3)
for i, name in enumerate(functional_inputs):
    with cols_f[i % 3]:
        def_val = get_default_val(name)
        is_checked = st.checkbox(f"{name}", value=bool(def_val > 0))
        user_vals[name] = 1.0 if is_checked else 0.0

# ==========================================
# 4. 幕后：百分百精确的硬映射合成
# ==========================================
st.divider()
final_row = {}

# 1. 直接填入存在的物理量
for col in continuous_inputs:
    if col in X_base.columns:
        final_row[col] = user_vals[col]

# 2. 直接填入官能团
for col in functional_inputs:
    if col in X_base.columns:
        final_row[col] = user_vals[col]

# 3. 严格执行 Log 对数转换
log_mapping = {
    'Log_adsorption time min': 'adsorption time min',
    'Log_specific surface area m2/g': 'specific surface area m2/g',
    'Log_initial concentration mg/L': 'initial concentration mg/L'
}
for log_col, raw_col in log_mapping.items():
    if log_col in X_base.columns:
        final_row[log_col] = np.log1p(user_vals[raw_col])

# 4. 严格执行派生比值计算 (分子分母绝不张冠李戴)
c0 = user_vals['initial concentration mg/L']
c0_safe = c0 if c0 != 0 else 1e-9

dose = user_vals['adsorbent dose mg/ml']
dose_safe = dose if dose != 0 else 1e-9

if 'Log_CA_to_C0_Ratio' in X_base.columns:
    final_row['Log_CA_to_C0_Ratio'] = np.log1p(user_vals['DOM_CA'] / c0_safe)

if 'Log_FA_to_C0_Ratio' in X_base.columns:
    final_row['Log_FA_to_C0_Ratio'] = np.log1p(user_vals['DOM_FA'] / c0_safe)

if 'Log_HA_to_C0_Ratio' in X_base.columns:
    final_row['Log_HA_to_C0_Ratio'] = np.log1p(user_vals['DOM_HA'] / c0_safe)

if 'Log_C0_to_Dose_Ratio' in X_base.columns:
    final_row['Log_C0_to_Dose_Ratio'] = np.log1p(c0 / dose_safe)

# 保底校验：确保 27 个特征全在这个字典里
for col in X_base.columns:
    if col not in final_row:
        final_row[col] = X_base[col].median()

# 对齐 DataFrame
predict_df = pd.DataFrame([final_row])[X_base.columns]

# ==========================================
# 5. 预测与展示
# ==========================================
try:
    res_log = models[selected_name].predict(predict_df)[0]
    res_real = np.expm1(res_log)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{res_real:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（后台已根据表头结构实现 100% 精确联动计算）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎链路异常: {e}")
