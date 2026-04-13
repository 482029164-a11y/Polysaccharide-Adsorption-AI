import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 🌟 0. 核心修正：补全类定义与 Scikit-learn 接口
# ==========================================

class StandardDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.network(x)

class PyTorchStandardRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
    def fit(self, X, y): return self # 必须具备此方法以通过 sklearn 校验
    def predict(self, X):
        self.model_.eval()
        dev = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Xt = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(dev)
        with torch.no_grad(): return self.model_(Xt).cpu().numpy().flatten()

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, k_ensembles=8, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.k_ensembles = k_ensembles
    def fit(self, X, y): return self
    def predict(self, X):
        dev = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Xt = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(dev)
        for m in self.models_: m.eval()
        with torch.no_grad(): return torch.cat([m(Xt) for m in self.models_], dim=1).mean(dim=1).cpu().numpy()

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.R = nn.Parameter(torch.ones(1, k_ensembles, input_dim) + torch.randn(1, k_ensembles, input_dim) * 0.01)
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.ReLU()
        )
        self.head_weights = nn.Parameter(torch.randn(k_ensembles, hidden_dim // 2) / math.sqrt(hidden_dim // 2))
        self.head_biases = nn.Parameter(torch.zeros(k_ensembles))
    def forward(self, x):
        x = x.unsqueeze(1) * self.R
        out = self.shared_bottom(x)
        return (out * self.head_weights).sum(dim=-1) + self.head_biases

class PyTorchTrueTabMRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
    def fit(self, X, y): return self # 核心修正
    def predict(self, X):
        self.model_.eval()
        dev = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Xt = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(dev)
        with torch.no_grad(): return self.model_(Xt).mean(dim=1).cpu().numpy()

# ==========================================
# 📥 加载数据
# ==========================================
@st.cache_resource
def load_data():
    return joblib.load('model_artifacts_v3.pkl')

data = load_data()
models = data['models']
X_cols = data['X'].columns.tolist()

# ==========================================
# 界面布局 (极简设计)
# ==========================================
st.title("多糖吸附量预测系统")

st.header("输入原始参数")
col1, col2 = st.columns(2)

user_raw_inputs = {}

with col1:
    st.subheader("材料与官能团")
    # 动态识别需要对数转换的物理属性
    user_raw_inputs['ssa'] = st.number_input("比表面积 (m²/g)", value=200.0)
    user_raw_inputs['mw'] = st.number_input("分子量 (kDa/MDa)", value=500.0)
    
    # 动态识别所有以 FG_ 开头的官能团特征
    fg_cols = [c for c in X_cols if c.startswith('FG_')]
    for fg in fg_cols:
        user_raw_inputs[fg] = st.checkbox(fg.replace('FG_', ''))

with col2:
    st.subheader("反应环境")
    user_raw_inputs['ph'] = st.number_input("pH", value=5.5)
    user_raw_inputs['is'] = st.number_input("离子强度 (mol/L)", value=0.01, format="%.4f")
    user_raw_inputs['time'] = st.number_input("吸附时间 (min)", value=120.0)
    user_raw_inputs['c0'] = st.number_input("初始浓度 C0 (mg/L)", value=50.0)
    user_raw_inputs['dose'] = st.number_input("投加量 Dose (mg/ml)", value=1.0)

st.subheader("DOM 浓度 (mg/L)")
# 动态识别所有以 DOM_ 开头的复合浓度特征
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
for dom in dom_cols:
    label = dom.replace('DOM_', '').replace('_浓度', '')
    user_raw_inputs[dom] = st.number_input(f"{label} 浓度", value=0.0)

# ==========================================
# 🚀 预测逻辑 (动态特征映射)
# ==========================================
if st.button("计算预测值"):
    # 建立最终的特征字典，确保与 X_cols 严格对应
    final_input_dict = {}

    # 1. 处理对数转换列 (严格同步训练脚本逻辑)
    if 'Log_specific surface area m2/g' in X_cols:
        final_input_dict['Log_specific surface area m2/g'] = np.log1p(user_raw_inputs['ssa'])
    if 'Log_molecular weight' in X_cols:
        final_input_dict['Log_molecular weight'] = np.log1p(user_raw_inputs['mw'])
    if 'Log_adsorption time min' in X_cols:
        final_input_dict['Log_adsorption time min'] = np.log1p(user_raw_inputs['time'])
    if 'Log_C0_to_Dose_Ratio' in X_cols:
        final_input_dict['Log_C0_to_Dose_Ratio'] = np.log1p(user_raw_inputs['c0'] / (user_raw_inputs['dose'] + 1e-5))
    
    # 2. 处理线性环境列
    if 'pH' in X_cols: final_input_dict['pH'] = user_raw_inputs['ph']
    if 'Ionic strength' in X_cols: final_input_dict['Ionic strength'] = user_raw_inputs['is']

    # 3. 处理官能团与 DOM 浓度 (这些列名在字典和 X_cols 中是一一对应的)
    for c in X_cols:
        if c in user_raw_inputs:
            final_input_dict[c] = float(user_raw_inputs[c])

    # 4. 转换成 DataFrame 并排序
    final_df = pd.DataFrame([final_input_dict]).reindex(columns=X_cols, fill_value=0.0)

    # 5. 执行预测
    best_name = 'True TabM' if 'True TabM' in models else list(models.keys())[0]
    
    try:
        pred_log = models[best_name].predict(final_df)[0]
        prediction = np.expm1(pred_log)
        
        st.markdown("---")
        st.metric(label=f"预测吸附量 Qm (mg/g)", value=f"{prediction:.2f}")
        st.caption(f"当前计算模型: {best_name}")
        
    except Exception as e:
        st.error(f"预测过程出错: {e}")
        st.info("排查提示：请确保 model_artifacts_v3.pkl 与此脚本在同一目录下。")
