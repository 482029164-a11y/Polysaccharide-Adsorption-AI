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
# 0. 必须存在的底层蓝图 (严密对齐内核)
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
# 1. 内核加载与全量特征分拣
# ==========================================
@st.cache_resource
def load_and_audit_features():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(f_path)
    torch.load = orig_load
    
    X_base = data['X']
    all_models = data['models']
    
    # 🔍 全量特征审计
    all_cols = X_base.columns.tolist()
    
    # 提取“根特征”（即用户需要填写的原始项）
    root_features = set()
    for col in all_cols:
        # 去掉派生修饰词
        temp = col.replace('Log_', '').replace('_Ratio', '').replace('Ratio', '').strip()
        # 特殊处理比值：如果包含 HA 和 C0，根特征应是 HA 和 C0 本身
        if 'HA_to_C0' in temp:
            root_features.add('DOM_HA')
            root_features.add('initial concentration mg/L')
        else:
            root_features.add(temp)

    # 细分根特征
    final_phys_ui = [] # 连续数值
    final_func_ui = [] # 勾选框
    
    for rf in sorted(list(root_features)):
        # 匹配原始数据中对应的真实列（用来算中位数）
        match_col = rf
        if rf not in all_cols:
            # 如果原名不在，找对应的 Log_ 名
            match_col = f"Log_{rf}" if f"Log_{rf}" in all_cols else None
        
        if match_col:
            unique_vals = X_base[match_col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                final_func_ui.append(rf)
            else:
                final_phys_ui.append(rf)
        else:
            # 如果依然匹配不到（如 Ratio 的拆分项），默认归为数值输入
            final_phys_ui.append(rf)

    return X_base, all_models, final_phys_ui, final_func_ui

X_base, models, phys_ui, func_ui = load_and_audit_features()

# ==========================================
# 2. 界面呈现
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统 (全量特征版)")

selected_model = st.selectbox("1. 选择模型引擎:", list(models.keys()))

st.divider()

# --- 数值输入区 ---
st.subheader("2. 物理与环境参数录入")
st.info("💡 默认值为中位数。Log 转换与 Ratio 合成已在后台自动激活。")
user_vals = {}
c1, c2 = st.columns(2)
for i, name in enumerate(phys_ui):
    with (c1 if i % 2 == 0 else c2):
        # 尝试寻找参考中位数
        ref_col = f"Log_{name}" if f"Log_{name}" in X_base.columns else name
        if ref_col in X_base.columns:
            m_val = np.expm1(X_base[ref_col].median()) if ref_col.startswith('Log_') else X_base[ref_col].median()
        else:
            m_val = 0.0
            
        user_vals[name] = st.number_input(f"{name}", value=float(m_val), format="%.4f")

st.divider()

# --- 勾选框区 ---
st.subheader("3. 表面官能团检测")
st.caption("勾选代表存在，不勾选代表不存在。")
cf1, cf2, cf3 = st.columns(3)
for i, name in enumerate(func_ui):
    with (cf1 if i % 3 == 0 else cf2 if i % 3 == 1 else cf3):
        is_checked = st.checkbox(f"{name}", value=False)
        user_vals[name] = 1.0 if is_checked else 0.0

# ==========================================
# 3. 幕后全量特征合成 (核心逻辑)
# ==========================================
st.divider()

final_row = {}

# 遍历模型需要的【每一个】特征，确保全部填充
for col in X_base.columns:
    # 逻辑 1：处理 Ratio (以 HA_to_C0 为例)
    if 'Ratio' in col:
        # 这里需要根据你实际的比值逻辑编写
        ha = user_vals.get('DOM_HA', 0)
        c0 = user_vals.get('initial concentration mg/L', 1e-9)
        ratio_val = ha / c0
        final_row[col] = np.log1p(ratio_val) if col.startswith('Log_') else ratio_val
    
    # 逻辑 2：处理普通的 Log 项
    elif col.startswith('Log_'):
        raw_name = col.replace('Log_', '')
        final_row[col] = np.log1p(user_vals.get(raw_name, 0))
        
    # 逻辑 3：处理原始项
    else:
        final_row[col] = user_vals.get(col, 0)

# 转换为 DataFrame 并对齐顺序
final_df = pd.DataFrame([final_row])[X_base.columns]

# ==========================================
# 4. 预测输出
# ==========================================
try:
    log_y = models[selected_model].predict(final_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{real_y:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（所有派生特征已完成实时同步）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎故障: {e}")
