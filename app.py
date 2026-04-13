import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 🌟 0. 核心修正：补全类定义并强制通过 sklearn 校验
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
        self.epochs = epochs; self.batch_size = batch_size; self.lr_min = lr_min; self.lr_max = lr_max; self.T_0 = T_0; self.T_mult = T_mult
    def __sklearn_is_fitted__(self): return True # 强制通过校验
    def predict(self, X):
        self.model_.eval()
        dev = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Xt = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(dev)
        with torch.no_grad(): return self.model_(Xt).cpu().numpy().flatten()

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, k_ensembles=8, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.k_ensembles = k_ensembles; self.epochs = epochs; self.batch_size = batch_size
    def __sklearn_is_fitted__(self): return True
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
        self.epochs = epochs; self.batch_size = batch_size; self.lr_min = lr_min; self.lr_max = lr_max; self.T_0 = T_0; self.T_mult = T_mult
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        self.model_.eval()
        dev = getattr(self, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        Xt = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(dev)
        with torch.no_grad(): return self.model_(Xt).mean(dim=1).cpu().numpy()

# ==========================================
# 📥 加载模型数据
# ==========================================
@st.cache_resource
def load_data():
    return joblib.load('model_artifacts_v3.pkl')

data = load_data()
models = data['models']
X_cols = data['X'].columns.tolist()

# ==========================================
# 🛠️ 界面布局 (自动扫描特征列)
# ==========================================
st.set_page_config(page_title="吸附量预测", layout="centered")
st.title("多糖基材料吸附量预测系统")

st.header("实验参数设置")
col1, col2 = st.columns(2)

# 存储用户输入的原始值
raw_inputs = {}

with col1:
    st.subheader("物理化学参数")
    # 这里我们定义一些基础的对应关系，如果模型里有这些 Log 特征，就显示输入框
    if any("surface area" in c for c in X_cols):
        raw_inputs['ssa'] = st.number_input("比表面积 (m²/g)", value=200.0)
    if any("molecular weight" in c for c in X_cols):
        raw_inputs['mw'] = st.number_input("分子量 (kDa/MDa)", value=500.0)
    if any("time" in c for c in X_cols):
        raw_inputs['time'] = st.number_input("吸附时间 (min)", value=120.0)
    
    raw_inputs['c0'] = st.number_input("初始浓度 C0 (mg/L)", value=50.0)
    raw_inputs['dose'] = st.number_input("投加量 Dose (mg/ml)", value=1.0)
    
    if 'pH' in X_cols:
        raw_inputs['pH'] = st.number_input("pH", value=5.5)
    if 'Ionic strength' in X_cols:
        raw_inputs['Ionic strength'] = st.number_input("离子强度 (mol/L)", value=0.01, format="%.4f")

with col2:
    st.subheader("官能团与 DOM")
    # 自动生成官能团 Checkbox
    fg_cols = [c for c in X_cols if c.startswith('FG_')]
    for fg in fg_cols:
        raw_inputs[fg] = st.checkbox(fg.replace('FG_', ''))
    
    # 自动生成 DOM 浓度输入框
    dom_cols = [c for c in X_cols if c.startswith('DOM_')]
    for dom in dom_cols:
        label = dom.replace('DOM_', '').replace('_浓度', '')
        raw_inputs[dom] = st.number_input(f"{label} 浓度 (mg/L)", value=0.0)

# ==========================================
# 🚀 推理逻辑 (特征自动转换与排序)
# ==========================================
if st.button("计算 Qm"):
    # 1. 严格按照 X_cols 的顺序构建输入数据
    final_input_row = {}
    
    for col in X_cols:
        # 对数转换类
        if col == 'Log_specific surface area m2/g':
            final_input_row[col] = np.log1p(raw_inputs.get('ssa', 0))
        elif col == 'Log_molecular weight':
            final_input_row[col] = np.log1p(raw_inputs.get('mw', 0))
        elif col == 'Log_adsorption time min':
            final_input_row[col] = np.log1p(raw_inputs.get('time', 0))
        elif col == 'Log_C0_to_Dose_Ratio':
            ratio = raw_inputs.get('c0', 0) / (raw_inputs.get('dose', 1) + 1e-5)
            final_input_row[col] = np.log1p(ratio)
        # 官能团和 DOM 浓度类 (直接对应列名)
        elif col in raw_inputs:
            final_input_row[col] = float(raw_inputs[col])
        # 其他数值类
        elif col == 'pH':
            final_input_row[col] = raw_inputs.get('pH', 5.5)
        elif col == 'Ionic strength':
            final_input_row[col] = raw_inputs.get('Ionic strength', 0.0)
        else:
            final_input_row[col] = 0.0 # 兜底补零

    # 2. 转换为 DataFrame
    df_input = pd.DataFrame([final_input_row])[X_cols]
    
    # 3. 运行预测
    try:
        # 优先使用 TabM
        best_model_name = 'True TabM' if 'True TabM' in models else list(models.keys())[0]
        model = models[best_model_name]
        
        # 模型输出的是 log1p(Qm)，需要反转
        log_pred = model.predict(df_input)[0]
        qm_pred = np.expm1(log_pred)
        
        st.divider()
        st.metric(label="预测吸附量 Qm (mg/g)", value=f"{qm_pred:.2f}")
        st.text(f"计算内核: {best_model_name}")
        
    except Exception as e:
        st.error(f"计算失败: {e}")
        st.info("排查建议：请确认类定义与模型训练脚本完全一致。")
