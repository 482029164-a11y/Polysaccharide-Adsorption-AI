import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
import sys
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 0. 必须存在的类定义 (必须包含 fit 方法以通过 sklearn 检查)
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
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / math.sqrt(hidden_dim // 2))
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        x = x.unsqueeze(1) * self.R
        out = self.shared_bottom(x)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

# --- 核心修正：补全评估器协议 ---
class PyTorchBaseWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=100, lr=0.001, batch_size=32):
        self.epochs = epochs; self.lr = lr; self.batch_size = batch_size
    def fit(self, X, y=None): return self # 必须存在此方法

class PyTorchSingleDNN(PyTorchBaseWrapper):
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device).eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        with torch.no_grad(): return self.model_(X_t).cpu().numpy().flatten()

class PyTorchDeepEnsemble(PyTorchBaseWrapper):
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        for m in self.models_: m.to(device).eval()
        with torch.no_grad():
            return torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()

class PyTorchTrueTabMRegressor(PyTorchBaseWrapper):
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'): self.model_.to(device).eval()
        X_t = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
        with torch.no_grad(): return self.model_(X_t).mean(dim=1).cpu().numpy().flatten()

# 类映射注册
sys.modules['__main__'].StandardDNN = StandardDNN
sys.modules['__main__'].TrueTabMMini = TrueTabMMini
sys.modules['__main__'].PyTorchSingleDNN = PyTorchSingleDNN
sys.modules['__main__'].PyTorchDeepEnsemble = PyTorchDeepEnsemble
sys.modules['__main__'].PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor

# ==========================================
# 1. 应用与模型加载
# ==========================================
st.set_page_config(page_title="Adsorption AI Predictor", layout="centered")
st.title("🧪 多糖吸附预测专家系统")

@st.cache_resource
def load_data():
    orig_load = torch.load
    torch.load = lambda *args, **kwargs: orig_load(*args, **kwargs, map_location='cpu')
    # 请确保你的文件名是 model_artifacts_final.pkl 或 model_artifacts_v32.pkl
    data = joblib.load('model_artifacts_final.pkl')
    torch.load = orig_load
    return data

try:
    kernel = load_data()
    X_base, models = kernel['X'], kernel['models']
except Exception as e:
    st.error(f"加载内核失败: {e}"); st.stop()

# ==========================================
# 2. 界面排版 (直接数值输入)
# ==========================================
st.subheader("1. 策略配置")
selected_name = st.selectbox("选择模型引擎:", list(models.keys()))

st.divider()
st.subheader("2. 特征参数手动录入")
st.info("💡 请在下方框内直接输入数值，默认值为数据集平均水平。")

user_input_dict = {}
cols = st.columns(2)
for i, col in enumerate(X_base.columns):
    with cols[i % 2]:
        # 获取数据集的统计边界供参考
        val_min = float(X_base[col].min())
        val_max = float(X_base[col].max())
        val_avg = float(X_base[col].mean())
        
        # 🌟 改为直接数值输入框 (number_input)
        user_input_dict[col] = st.number_input(
            f"{col}", 
            value=val_avg, 
            format="%.4f",
            help=f"参考范围: {val_min:.2f} ~ {val_max:.2f}"
        )

# ==========================================
# 3. 结果输出
# ==========================================
st.divider()
input_df = pd.DataFrame([user_input_dict])
active_model = models[selected_name]

# 核心预测
try:
    log_y = active_model.predict(input_df)[0]
    real_y = np.expm1(log_y)
    
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="margin:0; color:#1f77b4;">预测吸附量 Qm</h2>
        <h1 style="font-size:60px; margin:10px 0;">{real_y:.2f} <small style="font-size:20px;">mg/g</small></h1>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"预测计算出错: {e}")

# ==========================================
# 4. 可解释性 (SHAP)
# ==========================================
if st.checkbox("🔍 开启 SHAP 局部归因分析"):
    with st.spinner("计算中..."):
        try:
            if 'XGBoost' in selected_name or 'Random Forest' in selected_name:
                explainer = shap.TreeExplainer(active_model)
                shap_vals = explainer(input_df)
            else:
                def f(x): return active_model.predict(pd.DataFrame(x, columns=X_base.columns))
                explainer = shap.KernelExplainer(f, shap.kmeans(X_base, 5))
                shap_vals = explainer(input_df)
            
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_vals[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"当前模型暂不支持归因可视化: {e}")
