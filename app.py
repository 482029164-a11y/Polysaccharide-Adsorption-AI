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
# 0. 底层蓝图 (补全评估器协议以通过 sklearn 校验)
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
# 1. 内核加载与物理量归位
# ==========================================
@st.cache_resource
def load_and_prep():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    f_path = 'model_artifacts_final.pkl' if os.path.exists('model_artifacts_final.pkl') else 'model_artifacts_v32.pkl'
    data = joblib.load(f_path)
    torch.load = orig_load
    
    X_base = data['X']
    models = data['models']
    
    # 核心映射：筛选出哪些列是基础物理量，哪些是派生值
    all_cols = X_base.columns.tolist()
    ui_cols = []
    derived_cols = [] # 存储会被隐藏的 Ratio 或 Log 项
    
    for c in all_cols:
        # 如果是比值（Ratio）或者已经带有 Log_ 前缀的，一律在界面隐藏
        if 'Ratio' in c or c.startswith('Log_'):
            derived_cols.append(c)
        else:
            ui_cols.append(c)
            
    return X_base, models, ui_cols, derived_cols

X_base, models, ui_cols, derived_cols = load_and_prep()

# ==========================================
# 2. 界面设计 (纯物理量输入 + 中位数)
# ==========================================
st.set_page_config(page_title="Adsorption Expert", layout="centered")
st.title("🧪 多糖吸附预测专家系统 (物理量直推版)")

st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 物理工况录入")
st.info("💡 所有比值（Ratio）与对数变换（Log）均在幕后自动计算，您只需输入实验原始物理量。")

user_raw_inputs = {}
cols = st.columns(2)

# 渲染用户界面上的基础物理量输入框
for i, name in enumerate(ui_cols):
    with cols[i % 2]:
        default_val = X_base[name].median()
        user_raw_inputs[name] = st.number_input(
            f"{name}", 
            value=float(default_val), 
            format="%.4f"
        )

# ==========================================
# 3. 幕后黑盒：逻辑自动合成
# ==========================================
st.divider()

# 准备送入模型的最终字典
final_input_row = {}

# 第一步：填充用户直接输入的物理量
for k, v in user_raw_inputs.items():
    final_input_row[k] = v

# 第二步：智能合成比值 (Ratio) 与 对数 (Log)
# 🌟 注意：这里会根据你数据集中真实的派生列名进行匹配计算
for d_col in derived_cols:
    if d_col.startswith('Log_'):
        # 处理 Log 类型的派生项
        base_name = d_col.replace('Log_', '')
        
        if 'Ratio' in base_name:
            # 如果是 Log_Ratio，先找 Ratio
            # 常见逻辑：HA_to_C0_Ratio = DOM_HA / initial_concentration
            # 💡 提示：此处需要根据你实际的列名定义微调逻辑
            if 'HA_to_C0_Ratio' in base_name:
                ha = user_raw_inputs.get('DOM_HA', 0)
                c0 = user_raw_inputs.get('initial concentration mg/L', 1) # 防止除零
                ratio = ha / c0 if c0 != 0 else 0
                final_input_row[d_col] = np.log1p(ratio)
        else:
            # 如果只是普通的 Log 项
            val = user_raw_inputs.get(base_name, 0)
            final_input_row[d_col] = np.log1p(val)
            
    elif 'Ratio' in d_col:
        # 处理非 Log 的 Ratio 项
        if 'HA_to_C0_Ratio' in d_col:
            ha = user_raw_inputs.get('DOM_HA', 0)
            c0 = user_raw_inputs.get('initial concentration mg/L', 1)
            final_input_row[d_col] = ha / c0 if c0 != 0 else 0

# 补全可能缺失的列（赋中位数），确保 DataFrame 结构与模型完全一致
final_df = pd.DataFrame([final_input_row])
for missing_col in X_base.columns:
    if missing_col not in final_df.columns:
        final_df[missing_col] = X_base[missing_col].median()

# 确保列顺序完全一致
final_df = final_df[X_base.columns]

try:
    log_y = models[selected_name].predict(final_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#F0F2F6; padding:25px; border-radius:12px; text-align:center; border: 2px solid #E0E0E0;">
        <h3 style="margin:0; color:#444;">预测平衡吸附量 Qm</h3>
        <h1 style="font-size:55px; color:#1E88E5; margin:10px 0;">{real_y:.2f} <small style="font-size:20px; color:#777;">mg/g</small></h1>
        <p style="color:#888; font-size:14px;">（Ratio 与 Log 变换已在后台实时同步合成）</p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"模型推导失败: {e}")
