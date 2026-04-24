import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 必须存在的底层蓝图定义 (供内核读取)
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

import sys
sys.modules['__main__'].StandardDNN = StandardDNN
sys.modules['__main__'].TrueTabMMini = TrueTabMMini
sys.modules['__main__'].PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
sys.modules['__main__'].PyTorchSingleDNN = PyTorchSingleDNN
sys.modules['__main__'].PyTorchDeepEnsemble = PyTorchDeepEnsemble

# ==========================================
# 1. 核心应用与侧边栏配置
# ==========================================
st.set_page_config(page_title="AI-Driven Adsorption Predictor", layout="wide", page_icon="🧪")
st.title("🧪 复杂配体竞争体系下 $Q_m$ 预测专家系统")
st.markdown("基于 **深度特征寻优** 与 **自适应核密度逆向赋权 (IDW)** 构建。请在侧边栏调节材料和化学环境特征参数。")

@st.cache_resource
def load_kernel():
    # 🌟 核心补丁：Web端同样需要将显卡张量洗白为 CPU 张量
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, map_location='cpu')
    
    data = joblib.load('model_artifacts_final.pkl')
    
    torch.load = original_torch_load
    
    best_r2 = -float('inf')
    best_name_key = None
    for study_name, std in data['studies'].items():
        if std.best_value > best_r2:
            best_r2 = std.best_value
            best_name_key = study_name
    best_model_name = 'DNN Ensemble' if best_name_key == 'Deep_Ensemble' else best_name_key.replace('_', ' ')
    
    return data['X'], data['models'], best_model_name, best_r2

try:
    X_baseline, all_models, default_best_name, best_r2 = load_kernel()
except FileNotFoundError:
    st.error("🚨 无法找到内核文件！请确保 `model_artifacts_final.pkl` 存在于同一目录下。")
    st.stop()

# ==========================================
# 2. 交互式控制面板
# ==========================================
st.sidebar.header("🛠️ 智能模型与特征引擎设置")

selected_model_name = st.sidebar.selectbox(
    "1. 切换推断中枢 (Predictive Engine):", 
    list(all_models.keys()), 
    index=list(all_models.keys()).index(default_best_name)
)

st.sidebar.markdown(f"*当前模型全集交叉验证 OOF R²: **{best_r2:.4f}***")
st.sidebar.divider()
st.sidebar.subheader("2. 实时特征输入 (Real-time Features)")

input_data = {}
for col in X_baseline.columns:
    col_min = float(X_baseline[col].min())
    col_max = float(X_baseline[col].max())
    col_mean = float(X_baseline[col].mean())
    input_data[col] = st.sidebar.slider(
        col, 
        min_value=col_min, max_value=col_max, value=col_mean, 
        step=(col_max - col_min) / 100.0 if col_max > col_min else 1.0
    )

# ==========================================
# 3. 预测与可解释性解码
# ==========================================
user_df = pd.DataFrame([input_data])
active_model = all_models[selected_model_name]

log_pred = active_model.predict(user_df)[0]
real_qm = np.expm1(log_pred)

st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🎯 实时热力学响应结果")
    st.metric(label="Predicted Adsorption Capacity ($Q_m$)", value=f"{real_qm:.2f} mg/g", delta="Log_y: {:.3f}".format(log_pred))
    st.info(f"**模型状态**: 该推测受 {selected_model_name} 引擎及自适应惩罚区间约束。")

with col2:
    st.subheader("🔬 决策归因瀑布图 (Local SHAP Explanation)")
    with st.spinner("正在解码该工况下的模型决策路径..."):
        try:
            if 'XGBoost' in selected_model_name or 'Random Forest' in selected_model_name:
                explainer = shap.TreeExplainer(active_model)
                shap_values = explainer(user_df)
            else:
                def safe_predict(x_in):
                    if not isinstance(x_in, pd.DataFrame): x_in = pd.DataFrame(x_in, columns=user_df.columns)
                    return active_model.predict(x_in)
                background = shap.kmeans(X_baseline, 5)
                explainer = shap.KernelExplainer(safe_predict, background)
                shap_values = explainer(user_df)

            fig, ax = plt.subplots(figsize=(6, 4))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.warning(f"目前选中的深度架构暂不支持实时瀑布流解析。错误: {e}")
