import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 底层蓝图 (确保类定义完整以通过 sklearn 校验)
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
# 1. 内核加载与对数逻辑映射
# ==========================================
@st.cache_resource
def load_and_prep():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    # 自动探测文件
    file_name = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(file_name)
    torch.load = orig_load
    
    X_base = data['X']
    models = data['models']
    
    # 找出哪些列是带 Log_ 的
    log_cols = [c for c in X_base.columns if c.startswith('Log_')]
    raw_to_log = {c.replace('Log_', ''): c for c in log_cols}
    
    return X_base, models, raw_to_log

try:
    X_base, models, log_mapping = load_and_prep()
except Exception as e:
    st.error(f"加载内核失败: {e}"); st.stop()

# ==========================================
# 2. 界面显示 (数值输入 + 中位数填充)
# ==========================================
st.set_page_config(page_title="Adsorption AI", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 特征参数录入")
st.info("💡 默认填充值为数据集**中位数 (Median)**。系统将在幕后自动处理对数转换。")

user_raw_inputs = {}
cols = st.columns(2)

# 在界面上只显示非 Log 的原始名称
display_cols = []
for c in X_base.columns:
    display_name = c.replace('Log_', '')
    if display_name not in display_cols:
        display_cols.append(display_name)

for i, name in enumerate(display_cols):
    with cols[i % 2]:
        # 寻找对应的原始数据计算中位数
        actual_col = f"Log_{name}" if f"Log_{name}" in X_base.columns else name
        
        # 算对数还原后的中位数作为默认填充
        if actual_col.startswith("Log_"):
            median_val = np.expm1(X_base[actual_col].median())
        else:
            median_val = X_base[actual_col].median()
            
        user_raw_inputs[name] = st.number_input(
            f"{name}", 
            value=float(median_val), 
            format="%.4f"
        )

# ==========================================
# 3. 幕后黑盒转换与预测
# ==========================================
st.divider()
# 构造模型真正需要的 DataFrame
processed_input = {}
for col in X_base.columns:
    raw_name = col.replace('Log_', '')
    raw_value = user_raw_inputs[raw_name]
    
    if col.startswith('Log_'):
        # 🌟 幕后转换逻辑: Log1p 处理
        processed_input[col] = np.log1p(raw_value)
    else:
        processed_input[col] = raw_value

input_df = pd.DataFrame([processed_input])
active_model = models[selected_name]

try:
    log_y = active_model.predict(input_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="margin:0; color:#1f77b4;">预测吸附量 Qm</h2>
        <h1 style="font-size:60px; margin:10px 0;">{real_y:.2f} <small style="font-size:20px;">mg/g</small></h1>
        <p style="color:#666;">后台计算完成 (Inverse Log-transformed)</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测出错: {e}")
