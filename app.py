import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import sys
import os
import math
import optuna
import xgboost
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 0. 底层蓝图 (严密对齐训练内核)
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
# 1. 无损特征加载与参数沙箱
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
    
    display_features = {} 
    
    for col in X_base.columns:
        if 'Ratio' in col or 'ratio' in col.lower():
            display_features['dom_ha'] = 'DOM_HA'
            display_features['initial concentration mg/l'] = 'initial concentration mg/L'
            continue
            
        clean_name = col.replace('Log_', '').strip()
        lower_name = clean_name.lower()
        
        if lower_name not in display_features:
            display_features[lower_name] = clean_name

    ui_phys, ui_func = [], []
    
    for lower_name, clean_name in display_features.items():
        matched_cols = [c for c in X_base.columns if c.lower() == lower_name or c.lower() == f"log_{lower_name}"]
        if matched_cols:
            ref_col = matched_cols[0]
            unique_vals = X_base[ref_col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                ui_func.append(clean_name)
                continue
        ui_phys.append(clean_name)
            
    return X_base, models, sorted(ui_phys), sorted(ui_func)

try:
    X_base, models, ui_phys, ui_func = load_and_standardize()
except Exception as e:
    st.error(f"内核读取失败: {e}")
    st.stop()

# ==========================================
# 2. 界面设计 (引入 Form 机制)
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

selected_name = st.selectbox("1. 选择模型引擎:", list(models.keys()))

st.divider()

user_inputs = {}

# 🌟 核心改进：使用 st.form 隔离输入状态，阻止自动刷新
with st.form("expert_prediction_form"):
    st.subheader("2. 基础物理工况与官能团录入")
    st.info("💡 请自由调节下方参数。所有修改完成后，点击底部的【开始预测】按钮执行运算。")

    cols_p = st.columns(2)
    for i, name in enumerate(ui_phys):
        with cols_p[i % 2]:
            matched_cols = [c for c in X_base.columns if c.lower() == name.lower() or c.lower() == f"log_{name.lower()}"]
            if matched_cols:
                ref_col = matched_cols[0]
                m_val = X_base[ref_col].median()
                if ref_col.startswith("Log_"):
                    m_val = np.expm1(m_val)
            else:
                m_val = 0.0
            user_inputs[name] = st.number_input(f"{name}", value=float(m_val), format="%.4f")

    st.markdown("---")
    
    st.caption("表面官能团检测 (勾选代表存在)")
    cols_f = st.columns(3)
    for i, name in enumerate(ui_func):
        with cols_f[i % 3]:
            is_checked = st.checkbox(f"{name}", value=False)
            user_inputs[name] = 1.0 if is_checked else 0.0
            
    # 🌟 表单提交按钮
    submitted = st.form_submit_button("🚀 开始预测计算", use_container_width=True)

# ==========================================
# 3. 幕后智能合成与精确路由执行 (仅在点击按钮后执行)
# ==========================================
if submitted:
    user_inputs_lower = {k.lower(): v for k, v in user_inputs.items()}
    final_row = {}

    for col in X_base.columns:
        col_lower = col.lower().strip()
        
        # 精确路由：Ratio 处理
        if 'ratio' in col_lower:
            val = 0.0
            if 'ha_to_c0' in col_lower or 'ha/c0' in col_lower:
                num = user_inputs_lower.get('dom_ha', 0)
                den = user_inputs_lower.get('initial concentration mg/l', 1e-9)
                val = num / den if den != 0 else 0
            else:
                val = X_base[col].median()
                if col.startswith('Log_'):
                    val = np.expm1(val)
                print(f"⚠️ 警告: 发现未定义计算公式的比值特征 [{col}]，已默认填充中位数。")

            final_row[col] = np.log1p(val) if col.startswith('Log_') else val
            
        elif col.startswith('Log_'):
            root_lower = col.replace('Log_', '').strip().lower()
            val = user_inputs_lower.get(root_lower, 0)
            final_row[col] = np.log1p(val)
            
        else:
            root_lower = col.strip().lower()
            final_row[col] = user_inputs_lower.get(root_lower, X_base[col].median())

    predict_df = pd.DataFrame([final_row])[X_base.columns]

    # ==========================================
    # 4. 预测与展示
    # ==========================================
    try:
        res_log = models[selected_name].predict(predict_df)[0]
        res_real = np.expm1(res_log)
        
        st.markdown(f"""
        <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF; margin-top:20px;">
            <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
            <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{res_real:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
            <p style="color:#888; font-size:14px;">（特征降维处理与精确比值路由已合成完毕）</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"预测引擎链路异常: {e}")
