import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import sys
import os
import math
import optuna  # 🌟 必须导入 optuna，否则 joblib 无法解析 pkl 中的寻优对象
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 底层蓝图 (确保与内核逻辑完全闭环)
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
# 1. 终极安全内核加载与强力特征分拣
# ==========================================
@st.cache_resource
def load_and_standardize():
    # 🌟 终极安全沙箱拦截器：彻底防止 map_location 参数碰撞
    if not hasattr(torch, '_orig_load_backup'):
        torch._orig_load_backup = torch.load

    def safe_cpu_load(*args, **kwargs):
        # 直接暴力覆写字典中的键，绝不会引起 multiple values 报错
        kwargs['map_location'] = 'cpu'
        return torch._orig_load_backup(*args, **kwargs)

    torch.load = safe_cpu_load
    
    try:
        f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
        data = joblib.load(f_path)
    finally:
        # 无论成功还是失败，必定还原加载器，防止热重载崩溃
        torch.load = torch._orig_load_backup
    
    X_base = data['X']
    models = data['models']
    
    # --- 强力归一化分拣 ---
    display_to_actuals = {}
    
    for col in X_base.columns:
        if 'Ratio' in col or 'Log' in col:
            continue
            
        std_name = col.lower().strip()
        
        if std_name not in display_to_actuals:
            display_to_actuals[std_name] = []
        display_to_actuals[std_name].append(col)
        
    ui_phys = []
    ui_func = []
    
    for std_name, actuals in display_to_actuals.items():
        ref_col = actuals[0]
        unique_vals = X_base[ref_col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            ui_func.append(std_name)
        else:
            ui_phys.append(std_name)
            
    # 兜底确保 Ratio 联动所需的基础项存在
    for p in ['dom_ha', 'initial concentration mg/l']:
        if p not in ui_phys and p not in ui_func:
            ui_phys.append(p)
            
    return X_base, models, sorted(ui_phys), sorted(ui_func), display_to_actuals

try:
    X_base, models, ui_phys, ui_func, mapping = load_and_standardize()
except Exception as e:
    st.error(f"严重错误：内核读取失败。详细报错：{e}")
    st.stop()

# ==========================================
# 2. 交互界面设计
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 基础物理工况录入")
st.info("💡 已合并冗余特征并隐藏派生项。输入值将同步应用至所有隐藏维度。")

user_standard_inputs = {}
cols_p = st.columns(2)
for i, std_name in enumerate(ui_phys):
    with cols_p[i % 2]:
        actual_cols = mapping.get(std_name, [std_name])
        ref_col = actual_cols[0] if actual_cols[0] in X_base.columns else X_base.columns[0]
        
        m_val = X_base[ref_col].median() if ref_col in X_base.columns else 0.0
        user_standard_inputs[std_name] = st.number_input(f"{std_name.title()}", value=float(m_val), format="%.4f")

st.divider()
st.subheader("3. 表面官能团检测")
cols_f = st.columns(3)
for i, std_name in enumerate(ui_func):
    with cols_f[i % 3]:
        is_checked = st.checkbox(f"{std_name.title()}", value=False)
        user_standard_inputs[std_name] = 1.0 if is_checked else 0.0

# ==========================================
# 3. 幕后多点联动影子合成逻辑
# ==========================================
st.divider()

final_row = {}

for col in X_base.columns:
    col_std = col.lower().strip()
    
    if 'Ratio' in col:
        ha = user_standard_inputs.get('dom_ha', 0)
        c0 = user_standard_inputs.get('initial concentration mg/l', 1e-9)
        ratio = ha / c0
        final_row[col] = np.log1p(ratio) if 'Log' in col else ratio
        
    elif col.startswith('Log_'):
        root_std = col.replace('Log_', '').lower().strip()
        val = user_standard_inputs.get(root_std, 0)
        final_row[col] = np.log1p(val)
        
    elif col_std in user_standard_inputs:
        final_row[col] = user_standard_inputs[col_std]
        
    else:
        final_row[col] = X_base[col].median()

predict_df = pd.DataFrame([final_row])[X_base.columns]

# ==========================================
# 4. 预测推断与展示
# ==========================================
try:
    res_log = models[selected_name].predict(predict_df)[0]
    res_real = np.expm1(res_log)
    
    st.markdown(f"""
    <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{res_real:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（后台已自动同步合成对数变换与比值物理量）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎链路异常: {e}")
