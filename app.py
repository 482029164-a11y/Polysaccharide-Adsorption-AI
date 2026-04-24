import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import sys
import os
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 底层蓝图 (确保与训练内核 V32 严密对齐)
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
# 1. 深度内核加载与 UI 铁腕过滤
# ==========================================
@st.cache_resource
def load_and_purify_features():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    # 自动定位内核文件
    f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(f_path)
    torch.load = orig_load
    
    X_base = data['X']
    all_models = data['models']
    
    # --- 铁腕过滤逻辑 ---
    raw_physics = []
    functional_flags = []
    
    # 预定义的原子级物理量清单（优先级最高）
    atomic_primitives = ['DOM_HA', 'initial concentration mg/L', 'pH', 'temperature', 'adsorption time min']
    
    # 扫描所有列，提取真正的原始特征
    for col in X_base.columns:
        # 🌟 绝对禁止 Ratio 和 Log 字样出现在 UI 清单中
        if 'Ratio' in col or 'Log' in col:
            continue
            
        # 判断是勾选框还是数值框
        unique_vals = X_base[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            functional_flags.append(col)
        else:
            raw_physics.append(col)
            
    # 确保核心原子量一定存在
    for p in atomic_primitives:
        if p not in raw_physics: raw_physics.append(p)
            
    return X_base, all_models, sorted(list(set(raw_physics))), sorted(list(set(functional_flags)))

X_base, models, ui_phys, ui_func = load_and_purify_features()

# ==========================================
# 2. 交互式界面设计
# ==========================================
st.set_page_config(page_title="Adsorption AI", layout="centered")
st.title("🧪 多糖吸附预测专家系统 (V32 纯净版)")

selected_model = st.selectbox("1. 选择预测中枢:", list(models.keys()))

st.divider()

# --- 数值输入区 ---
st.subheader("2. 基础物理工况")
st.info("💡 所有对数(Log)与比值(Ratio)已隐藏，由后台自动合成。默认值为中位数。")
user_vals = {}
p_col1, p_col2 = st.columns(2)
for i, name in enumerate(ui_phys):
    with (p_col1 if i % 2 == 0 else p_col2):
        # 寻找参考中位数
        ref = name
        if ref not in X_base.columns:
            ref = f"Log_{name}" if f"Log_{name}" in X_base.columns else X_base.columns[0]
        
        m_val = np.expm1(X_base[ref].median()) if ref.startswith('Log_') else X_base[ref].median()
        user_vals[name] = st.number_input(f"{name}", value=float(m_val), format="%.4f")

st.divider()

# --- 官能团勾选区 ---
st.subheader("3. 表面官能团 (勾选即存在)")
f_col1, f_col2, f_col3 = st.columns(3)
for i, name in enumerate(ui_func):
    with (f_col1 if i % 3 == 0 else f_col2 if i % 3 == 1 else f_col3):
        is_on = st.checkbox(f"{name}", value=False)
        user_vals[name] = 1.0 if is_on else 0.0

# ==========================================
# 3. 幕后黑盒：影子合成逻辑
# ==========================================
st.divider()

final_input = {}

# 遍历模型所需的每一个特征列（包括那些在 UI 隐藏的）
for col in X_base.columns:
    # 1. 如果该列是 UI 上的物理量或勾选框，直接取值
    if col in user_vals:
        final_input[col] = user_vals[col]
        
    # 2. 处理复杂的派生比值 (Ratio)
    elif 'Ratio' in col:
        # 自动识别 HA / C0 的比值逻辑
        ha = user_vals.get('DOM_HA', 0)
        c0 = user_vals.get('initial concentration mg/L', 1e-9) # 防除零
        ratio = ha / c0
        final_input[col] = np.log1p(ratio) if 'Log' in col else ratio
        
    # 3. 处理普通的 Log 项
    elif col.startswith('Log_'):
        raw_key = col.replace('Log_', '')
        final_input[col] = np.log1p(user_vals.get(raw_key, 0))
        
    # 4. 保底：如果还没匹配上，给个中位数
    else:
        final_input[col] = X_base[col].median()

# 对齐 DataFrame 结构
predict_df = pd.DataFrame([final_input])[X_base.columns]

# ==========================================
# 4. 预测呈现
# ==========================================
try:
    log_res = models[selected_model].predict(predict_df)[0]
    final_qm = np.expm1(log_res)
    
    st.markdown(f"""
    <div style="background-color:#E3F2FD; padding:30px; border-radius:15px; text-align:center; border:2px solid #2196F3;">
        <h3 style="margin:0; color:#1565C0;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:65px; color:#0D47A1; margin:10px 0;">{final_qm:.2f} <small style="font-size:22px;">mg/g</small></h1>
        <p style="color:#546E7A; font-size:14px;">（后台已根据输入自动合成 {len(X_base.columns)} 个高阶特征）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎同步失败: {e}")
