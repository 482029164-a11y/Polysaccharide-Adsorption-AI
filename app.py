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
# 0. 必须存在的底层蓝图定义 (与内核 V32 严密对齐)
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
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / 11.31) # sqrt(128)
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

# 强制注册类
sys.modules['__main__'].StandardDNN = StandardDNN
sys.modules['__main__'].TrueTabMMini = TrueTabMMini
sys.modules['__main__'].PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
sys.modules['__main__'].PyTorchSingleDNN = PyTorchSingleDNN
sys.modules['__main__'].PyTorchDeepEnsemble = PyTorchDeepEnsemble

# ==========================================
# 1. 网页配置
# ==========================================
st.set_page_config(page_title="Adsorption AI Predictor", layout="centered", page_icon="🧪")
st.title("🧪 多糖吸附预测专家系统")

@st.cache_resource
def load_kernel():
    # 物理洗白显卡张量至 CPU
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, map_location='cpu')
    data = joblib.load('model_artifacts_final.pkl')
    torch.load = original_torch_load
    
    # 动态匹配最佳模型
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
# 2. 输入区域 (改为直接输入数值)
# ==========================================
st.subheader("1. 模型与中枢设置")
selected_model = st.selectbox("选择推断引擎:", list(all_models.keys()), index=list(all_models.keys()).index(default_best))

st.divider()
st.subheader("2. 特征参数手动输入")
st.info("💡 请直接在下方框内输入数值，框内的默认值为数据集平均值。")

input_values = {}
# 将特征分两列排列，视觉上更紧凑
cols = st.columns(2)
for i, col_name in enumerate(X_baseline.columns):
    with cols[i % 2]:
        val_min = float(X_baseline[col_name].min())
        val_max = float(X_baseline[col_name].max())
        val_mean = float(X_baseline[col_name].mean())
        
        # 🌟 核心修改：使用 number_input 替代 slider
        input_values[col_name] = st.number_input(
            f"{col_name}", 
            min_value=None,  # 不强制设限以增加灵活性，如需限制可改为 val_min
            max_value=None, 
            value=val_mean,
            format="%.4f"    # 保留 4 位小数
        )

# ==========================================
# 3. 计算预测结果
# ==========================================
st.divider()
user_input = pd.DataFrame([input_values])
pred_log = all_models[selected_model].predict(user_input)[0]
pred_real = np.expm1(pred_log)

# 使用大字号突出显示结果
st.markdown(f"""
<div style="background-color:#F0F2F6;border-radius:10px;padding:20px;text-align:center;">
    <h3 style="color:#1F77B4;margin:0;">预测平衡吸附量 Qm</h3>
    <h1 style="font-size:50px;margin:10px 0;">{pred_real:.2f} <small style="font-size:20px;">mg/g</small></h1>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. 可选的可解释性分析
# ==========================================
st.write("")
if st.checkbox("🔍 开启局部 SHAP 归因分析 (显示各参数贡献度)"):
    with st.spinner("正在解码黑盒决策路径..."):
        try:
            model_obj = all_models[selected_model]
            if 'XGBoost' in selected_model or 'Random Forest' in selected_model:
                explainer = shap.TreeExplainer(model_obj)
                shap_vals = explainer(user_input)
            else:
                def p(x): return model_obj.predict(pd.DataFrame(x, columns=X_baseline.columns))
                explainer = shap.KernelExplainer(p, shap.kmeans(X_baseline, 5))
                shap_vals = explainer(user_input)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"该模型的 SHAP 解释暂不可用: {e}")
