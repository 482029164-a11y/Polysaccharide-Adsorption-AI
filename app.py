import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 必须存在的底层蓝图 (补全评估器协议)
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
# 1. 内核加载与智能 UI 映射
# ==========================================
@st.cache_resource
def load_and_map():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(f_path)
    torch.load = orig_load
    
    X_base = data['X']
    models = data['models']
    
    # 【核心逻辑】：识别显示列（去重，去掉 Log_ 前缀）
    all_cols = X_base.columns.tolist()
    ui_display_cols = []
    for c in all_cols:
        clean_name = c.replace('Log_', '')
        if clean_name not in ui_display_cols:
            ui_display_cols.append(clean_name)
    
    return X_base, models, ui_display_cols

X_base, models, ui_cols = load_and_map()

# ==========================================
# 2. 界面设计 (纯数值输入 + 中位数)
# ==========================================
st.set_page_config(page_title="Adsorption AI", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 特征参数手动录入")
st.info("💡 系统已自动归并 Log 特征。请直接输入原始数值，默认值为中位数。")

user_raw_inputs = {}
cols = st.columns(2)

for i, name in enumerate(ui_cols):
    with cols[i % 2]:
        # 寻找该特征对应的真实列名（可能是原名，也可能是 Log_ 名）
        actual_col = f"Log_{name}" if f"Log_{name}" in X_base.columns else name
        
        # 自动反求原始值的中位数
        if actual_col.startswith("Log_"):
            default_val = np.expm1(X_base[actual_col].median())
        else:
            default_val = X_base[actual_col].median()
            
        user_raw_inputs[name] = st.number_input(
            f"{name}", 
            value=float(default_val), 
            format="%.4f"
        )

# ==========================================
# 3. 幕后自动转换与预测
# ==========================================
st.divider()

# 核心：根据用户输入的原始值，自动构造模型需要的 [Raw, Log] 混合特征行
final_input_dict = {}
for col in X_base.columns:
    base_name = col.replace('Log_', '')
    raw_val = user_raw_inputs[base_name]
    
    if col.startswith('Log_'):
        final_input_dict[col] = np.log1p(raw_val) # 幕后自动算对数
    else:
        final_input_dict[col] = raw_val

input_df = pd.DataFrame([final_input_dict])

try:
    log_y = models[selected_name].predict(input_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#f8f9fa; padding:20px; border-radius:10px; border:1px solid #dee2e6; text-align:center;">
        <h3 style="margin:0; color:#495057;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:55px; color:#007BFF; margin:10px 0;">{real_y:.2f} <small style="font-size:20px; color:#6c757d;">mg/g</small></h1>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"计算失败: {e}")
