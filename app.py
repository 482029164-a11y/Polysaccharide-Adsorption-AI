import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 底层蓝图 (确保与内核 V32 严格对齐)
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
            # 修正：根据您之前的代码逻辑，这里通常是 LayerNorm
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
# 1. 智能特征分类与内核载入
# ==========================================
@st.cache_resource
def load_and_classify():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(f_path)
    torch.load = orig_load
    
    X_base = data['X']
    models = data['models']
    
    # 🌟 特征自动化分拣逻辑
    continuous_physics = []  # 基础物理量（SSA, Pore size等）
    functional_groups = []   # 官能团（0/1 特征）
    derived_cols = []        # 幕后派生项（Log, Ratio）

    for col in X_base.columns:
        if 'Ratio' in col or col.startswith('Log_'):
            derived_cols.append(col)
        else:
            # 判断是否为官能团：检查数据是否只包含 0 和 1
            unique_vals = X_base[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                functional_groups.append(col)
            else:
                continuous_physics.append(col)
                
    return X_base, models, continuous_physics, functional_groups, derived_cols

X_base, models, phys_cols, func_cols, derived_cols = load_and_classify()

# ==========================================
# 2. 界面设计 (数值输入 + 官能团打勾)
# ==========================================
st.set_page_config(page_title="Adsorption Expert AI", layout="centered")
st.title(" 多糖吸附预测")

st.subheader("1. 推断引擎配置")
selected_name = st.selectbox("选择模型中枢:", list(models.keys()))

st.divider()

# --- 物理量输入区 ---
st.info("💡 请输入实验原始物理量，默认填充为中位数。")
user_inputs = {}
cols_phys = st.columns(2)
for i, name in enumerate(phys_cols):
    with cols_phys[i % 2]:
        user_inputs[name] = st.number_input(f"{name}", value=float(X_base[name].median()), format="%.4f")

st.divider()

# --- 官能团勾选区 ---
st.subheader("3. 表面官能团/组分检测")
st.caption("勾选代表【存在/已修饰】，不勾选默认【不存在/未检出】。")
cols_func = st.columns(3) # 官能团分三列显示，更节省空间
for i, name in enumerate(func_cols):
    with cols_func[i % 3]:
        # 🌟 核心改进：使用 checkbox 替代 number_input
        checked = st.checkbox(f"{name}", value=False)
        user_inputs[name] = 1.0 if checked else 0.0

# ==========================================
# 3. 幕后黑盒：智能数据合成
# ==========================================
st.divider()

# 最终送入模型的特征行
final_row = {}

# 1. 填充 UI 上的所有输入（物理量 + 官能团转化后的 0/1）
for k, v in user_inputs.items():
    final_row[k] = v

# 2. 自动合成隐藏的 Ratio 和 Log 项
for d_col in derived_cols:
    if d_col.startswith('Log_'):
        base_name = d_col.replace('Log_', '')
        
        # 处理可能的 Log_Ratio 情况
        if 'Ratio' in base_name:
            if 'HA_to_C0_Ratio' in base_name:
                ha = user_inputs.get('DOM_HA', 0)
                c0 = user_inputs.get('initial concentration mg/L', 1e-9)
                final_row[d_col] = np.log1p(ha / c0)
        else:
            val = user_inputs.get(base_name, 0)
            final_row[d_col] = np.log1p(val)
            
    elif 'Ratio' in d_col:
        # 处理普通的 Ratio 情况
        if 'HA_to_C0_Ratio' in d_col:
            ha = user_inputs.get('DOM_HA', 0)
            c0 = user_inputs.get('initial concentration mg/L', 1e-9)
            final_row[d_col] = ha / c0

# 补全可能缺失的列并对齐顺序
final_df = pd.DataFrame([final_row])
for m_col in X_base.columns:
    if m_col not in final_df.columns:
        final_df[m_col] = X_base[m_col].median()
final_df = final_df[X_base.columns]

# ==========================================
# 4. 预测与结果呈现
# ==========================================
try:
    log_y = models[selected_name].predict(final_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#F8F9FA; padding:25px; border-radius:15px; text-align:center; border: 2px solid #007BFF;">
        <h3 style="margin:0; color:#555;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{real_y:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#999; font-size:13px;">（后台已自动完成对数变换与比值联动计算）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测过程出现异常: {e}")
