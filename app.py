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

# 🌟 修复1：页面配置必须作为第一条 Streamlit 指令
st.set_page_config(page_title="Adsorption Expert", layout="centered")

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
# 1. 安全内核加载与【全局统一特征注册表】
# ==========================================
@st.cache_resource
def load_and_standardize():
    if not hasattr(torch, '_orig_load_backup'):
        torch._orig_load_backup = torch.load

    def safe_cpu_load(*args, **kwargs):
        kwargs.pop('map_location', None)
        if len(args) >= 2:
            args_list = list(args)
            args_list[1] = 'cpu'
            args = tuple(args_list)
        else:
            kwargs['map_location'] = 'cpu'
        return torch._orig_load_backup(*args, **kwargs)

    torch.load = safe_cpu_load
    try:
        f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
        data = joblib.load(f_path)
    finally:
        torch.load = torch._orig_load_backup
    
    X_base = data['X']
    models = data['models']
    
    ui_registry = {}
    
    for col in X_base.columns:
        if 'ratio' in col.lower():
            continue 
            
        clean_name = col.replace('Log_', '').strip()
        lower_id = clean_name.lower()
        
        if lower_id not in ui_registry:
            is_bool = set(X_base[col].dropna().unique()).issubset({0, 1})
            median_val = X_base[col].median()
            if col.startswith('Log_'):
                median_val = np.expm1(median_val)
                
            ui_registry[lower_id] = {
                'display_name': clean_name,
                'is_bool': is_bool,
                'default': median_val
            }
        else:
            if not col.startswith('Log_'):
                is_bool = set(X_base[col].dropna().unique()).issubset({0, 1})
                ui_registry[lower_id]['is_bool'] = is_bool
                ui_registry[lower_id]['default'] = X_base[col].median()

    for p_id, p_disp in [('dom_ha', 'DOM_HA'), ('initial concentration mg/l', 'initial concentration mg/L')]:
        if p_id not in ui_registry:
            ui_registry[p_id] = {'display_name': p_disp, 'is_bool': False, 'default': 1.0}

    ui_phys = {k: v for k, v in ui_registry.items() if not v['is_bool']}
    ui_func = {k: v for k, v in ui_registry.items() if v['is_bool']}
            
    return X_base, models, ui_phys, ui_func

try:
    X_base, models, ui_phys, ui_func = load_and_standardize()
except Exception as e:
    st.error(f"内核读取失败: {e}")
    st.stop()

# ==========================================
# 2. 交互界面设计 (恢复实时响应)
# ==========================================
st.title("🧪 多糖吸附预测专家系统")

selected_name = st.selectbox("1. 选择模型引擎:", list(models.keys()))
st.divider()

st.subheader("2. 基础物理工况录入")
st.info("💡 参数修改后，结果将实现毫秒级实时刷新。")

user_inputs = {}

cols_p = st.columns(2)
idx = 0
for lower_id, info in ui_phys.items():
    with cols_p[idx % 2]:
        user_inputs[lower_id] = st.number_input(
            f"{info['display_name']}", 
            value=float(info['default']), 
            format="%.4f"
        )
    idx += 1

st.markdown("---")
st.subheader("3. 表面官能团检测")
st.caption("勾选代表存在。")

cols_f = st.columns(3)
idx = 0
for lower_id, info in ui_func.items():
    with cols_f[idx % 3]:
        is_checked = st.checkbox(f"{info['display_name']}", value=bool(info['default'] > 0))
        user_inputs[lower_id] = 1.0 if is_checked else 0.0
    idx += 1

# ==========================================
# 3. 幕后：绝对严谨的合成重构
# ==========================================
st.divider()
final_row = {}

for col in X_base.columns:
    col_lower = col.lower().strip()
    
    # [1] 处理派生比值
    if 'ratio' in col_lower:
        val = 0.0
        # 🌟 修复2：放宽了比值判定规则，防止特征名称未能严格匹配导致联动静默失效
        if ('ha' in col_lower and 'c0' in col_lower) or 'ha_to_c0' in col_lower:
            ha = user_inputs.get('dom_ha', 1.0)
            c0 = user_inputs.get('initial concentration mg/l', 1.0)
            val = ha / c0 if c0 != 0 else 0.0
        else:
            val = X_base[col].median()
            if col.startswith('Log_'):
                val = np.expm1(val)
                
        final_row[col] = np.log1p(val) if col.startswith('Log_') else val
        
    # [2] 处理对数特征
    elif col.startswith('Log_'):
        root_id = col.replace('Log_', '').strip().lower()
        val = user_inputs.get(root_id, 0.0)
        final_row[col] = np.log1p(val)
        
    # [3] 处理原始物理特征
    else:
        root_id = col.strip().lower()
        final_row[col] = user_inputs.get(root_id, X_base[col].median())

predict_df = pd.DataFrame([final_row])[X_base.columns]

# ==========================================
# 4. 预测与展示
# ==========================================
try:
    res_log = models[selected_name].predict(predict_df)[0]
    res_real = np.expm1(res_log)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{res_real:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（数据已实时同步运算并精确路由至 {len(X_base.columns)} 维特征矩阵）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎链路异常: {e}")
