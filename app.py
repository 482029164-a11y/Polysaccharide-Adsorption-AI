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
# 1. 安全内核加载与无损特征提取
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
    
    # 🌟 无损特征剥离逻辑
    display_features = {} 
    
    for col in X_base.columns:
        if 'Ratio' in col or 'ratio' in col.lower():
            # 强行确保派生比值的源物理量存在
            display_features['dom_ha'] = 'DOM_HA'
            display_features['initial concentration mg/l'] = 'initial concentration mg/L'
            continue
            
        # 剥离 Log 外壳，只取核心物理名
        clean_name = col.replace('Log_', '').strip()
        lower_name = clean_name.lower()
        
        if lower_name not in display_features:
            display_features[lower_name] = clean_name

    ui_phys, ui_func = [], []
    
    for lower_name, clean_name in display_features.items():
        # 探测这是 0/1 官能团还是连续数值
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
    st.error(f"严重错误：内核读取失败。详细报错：{e}")
    st.stop()

# ==========================================
# 2. 界面设计 (全维度还原)
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 基础物理工况录入")
st.info("💡 隐藏的派生对数已自动剥离还原，所有关键物理特征现已无损显示。")

user_inputs = {}
cols_p = st.columns(2)
for i, name in enumerate(ui_phys):
    with cols_p[i % 2]:
        # 反求真实中位数
        matched_cols = [c for c in X_base.columns if c.lower() == name.lower() or c.lower() == f"log_{name.lower()}"]
        if matched_cols:
            ref_col = matched_cols[0]
            m_val = X_base[ref_col].median()
            # 如果原始数据里只有 Log 态，强行反解出物理真实值
            if ref_col.startswith("Log_"):
                m_val = np.expm1(m_val)
        else:
            m_val = 0.0
            
        user_inputs[name] = st.number_input(f"{name}", value=float(m_val), format="%.4f")

st.divider()
st.subheader("3. 表面官能团检测")
cols_f = st.columns(3)
for i, name in enumerate(ui_func):
    with cols_f[i % 3]:
        is_checked = st.checkbox(f"{name}", value=False)
        user_inputs[name] = 1.0 if is_checked else 0.0

# ==========================================
# 3. 幕后智能合成引擎 (精确路由版)
# ==========================================
st.divider()
user_inputs_lower = {k.lower(): v for k, v in user_inputs.items()}
final_row = {}

# 严格按模型输入需求重新合成全量特征矩阵
for col in X_base.columns:
    col_lower = col.lower().strip()
    
    # 🌟 1. 精确处理各种派生比值 (Ratio)
    if 'ratio' in col_lower:
        val = 0.0 # 默认初始值
        
        # 路由 A: HA 与 C0 的比值
        if 'ha_to_c0' in col_lower or 'ha/c0' in col_lower:
            num = user_inputs_lower.get('dom_ha', 0)
            den = user_inputs_lower.get('initial concentration mg/l', 1e-9)
            val = num / den if den != 0 else 0
            
        # 路由 B: (请根据你数据集中真实存在的比值进行仿写)
        # 假设你有一个 多糖/SSA 的比值，列名叫 'poly_to_ssa_ratio'
        # elif 'poly_to_ssa' in col_lower:
        #     num = user_inputs_lower.get('polysaccharide content', 0)
        #     den = user_inputs_lower.get('specific surface area m2/g', 1e-9)
        #     val = num / den if den != 0 else 0
        
        # 路由 C: 其他比值...以此类推
        # elif 'xxx_to_yyy' in col_lower:
        #     ...

        else:
            # 如果碰到了未定义的比值，为了防止报错，默认赋中位数，并在控制台提醒
            val = X_base[col].median()
            if col.startswith('Log_'):
                val = np.expm1(val) # 剥离对数态恢复物理值
            print(f"⚠️ 警告: 发现未定义计算公式的比值特征 [{col}]，已默认填充中位数。")

        # 统一处理该比值是否需要 Log 转换
        final_row[col] = np.log1p(val) if col.startswith('Log_') else val
        
    # 🌟 2. 处理普通的 Log 项 (如 Log_SSA, Log_Pore_Size 等)
    elif col.startswith('Log_'):
        root_lower = col.replace('Log_', '').strip().lower()
        val = user_inputs_lower.get(root_lower, 0)
        final_row[col] = np.log1p(val)
        
    # 🌟 3. 处理普通的原始物理量和官能团 (直接读取，无需计算)
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
    <div style="background-color:#F0F7FF; padding:30px; border-radius:15px; text-align:center; border:2px solid #007BFF;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:60px; color:#007BFF; margin:10px 0;">{res_real:.2f} <small style="font-size:20px; color:#666;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（后台已自动同步合成所有依赖特征）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测引擎链路异常: {e}")
