import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import sys
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 必须存在的底层蓝图定义 (严密对齐内核)
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
    def __init__(self, input_dim, hidden_dim=128, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.R = nn.Parameter(torch.ones(1, k_ensembles, input_dim) + torch.randn(1, k_ensembles, input_dim) * 0.01)
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1))
        )
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / math.sqrt(hidden_dim // 2))
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        x = x.unsqueeze(1) * self.R
        out = self.shared_bottom(x)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

class PyTorchTrueTabMRegressor(BaseEstimator, RegressorMixin):
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        with torch.no_grad(): return self.model_(X_t).mean(dim=1).cpu().numpy().flatten()

class PyTorchSingleDNN(BaseEstimator, RegressorMixin):
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        with torch.no_grad(): return self.model_(X_t).cpu().numpy().flatten()

class PyTorchDeepEnsemble(BaseEstimator, RegressorMixin):
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32).to(device)
        for m in self.models_: m.to(device); m.eval()
        with torch.no_grad():
            return torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()

# 强制注册类到 __main__ 防止 joblib 加载失败
sys.modules['__main__'].StandardDNN = StandardDNN
sys.modules['__main__'].TrueTabMMini = TrueTabMMini
sys.modules['__main__'].PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
sys.modules['__main__'].PyTorchSingleDNN = PyTorchSingleDNN
sys.modules['__main__'].PyTorchDeepEnsemble = PyTorchDeepEnsemble

# ==========================================
# 1. 应用配置 (界面还原至主屏)
# ==========================================
st.set_page_config(page_title="Qm Predictor", layout="centered", page_icon="🧪")
st.title("🧪 多糖吸附预测专家系统 (修正版)")

@st.cache_resource
def load_kernel():
    # 物理洗白显卡张量
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, map_location='cpu')
    data = joblib.load('model_artifacts_final.pkl')
    torch.load = original_torch_load
    
    # 获取寻优历史中的最佳模型名
    best_r2 = -float('inf')
    best_key = "XGBoost"
    for k, v in data['studies'].items():
        if v.best_value > best_r2:
            best_r2 = v.best_value
            best_key = k
    best_model_name = 'DNN Ensemble' if best_key == 'Deep_Ensemble' else best_key.replace('_', ' ')
    return data['X'], data['models'], best_model_name

X_baseline, all_models, default_best = load_kernel()

# ==========================================
# 2. 主界面输入区域 (还原排版)
# ==========================================
st.subheader("1. 模型选择")
selected_model = st.selectbox("选择推断引擎:", list(all_models.keys()), index=list(all_models.keys()).index(default_best))

st.divider()
st.subheader("2. 特征参数调节")

# 在主界面平铺输入滑块
input_values = {}
cols = st.columns(2) # 分两列排版更整齐
for i, col_name in enumerate(X_baseline.columns):
    with cols[i % 2]:
        val_min = float(X_baseline[col_name].min())
        val_max = float(X_baseline[col_name].max())
        val_mean = float(X_baseline[col_name].mean())
        input_values[col_name] = st.slider(f"{col_name}", val_min, val_max, val_mean)

# ==========================================
# 3. 预测输出
# ==========================================
st.divider()
user_input = pd.DataFrame([input_values])
pred_log = all_models[selected_model].predict(user_input)[0]
pred_real = np.expm1(pred_log)

st.success(f"### 预测吸附量 $Q_m$: **{pred_real:.2f}** mg/g")

# ==========================================
# 4. SHAP 局部解释 (放在底部)
# ==========================================
if st.checkbox("查看模型决策归因 (SHAP)"):
    with st.spinner("正在计算贡献度..."):
        try:
            model_obj = all_models[selected_model]
            if 'XGBoost' in selected_model or 'Random Forest' in selected_model:
                explainer = shap.TreeExplainer(model_obj)
                shap_vals = explainer(user_input)
            else:
                def p(x): return model_obj.predict(pd.DataFrame(x, columns=X_baseline.columns))
                explainer = shap.KernelExplainer(p, shap.kmeans(X_baseline, 5))
                shap_vals = explainer(user_input)
            
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"解释图生成失败: {e}")
