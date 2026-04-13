import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 类定义 (必须与训练脚本完全一致，确保模型加载)
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
    def predict(self, X):
        self.model_.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad(): return self.model_(X_t).cpu().numpy().flatten()

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    def predict(self, X):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: m.eval()
        with torch.no_grad():
            return torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy()

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
    def predict(self, X):
        self.model_.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            return self.model_(X_t).mean(dim=1).cpu().numpy()

# ==========================================
# 2. 加载数据
# ==========================================
@st.cache_resource
def load_data():
    return joblib.load('model_artifacts_v3.pkl')

data = load_data()
models = data['models']
X_cols = data['X'].columns.tolist()

# ==========================================
# 3. 界面布局
# ==========================================
st.set_page_config(page_title="吸附预测系统", layout="centered")
st.title("多糖基材料吸附量预测系统")

# 参数录入
st.header("输入实验参数")

col1, col2 = st.columns(2)

with col1:
    st.subheader("材料属性")
    raw_ssa = st.number_input("比表面积 (m²/g)", value=200.0)
    raw_mw = st.number_input("分子量 (kDa/MDa)", value=500.0)
    
    st.subheader("官能团 (勾选表示存在)")
    fg_cols = [c for c in X_cols if c.startswith('FG_')]
    user_fgs = {fg: st.checkbox(fg.replace('FG_', '')) for fg in fg_cols}

with col2:
    st.subheader("反应条件")
    raw_ph = st.number_input("pH", value=5.5)
    raw_is = st.number_input("离子强度 (mol/L)", value=0.01, format="%.4f")
    raw_time = st.number_input("吸附时间 (min)", value=120.0)
    raw_c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0)
    raw_dose = st.number_input("投加量 Dose (mg/ml)", value=1.0)

st.subheader("DOM 浓度 (mg/L)")
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
user_doms = {dom: st.number_input(dom.replace('DOM_', '').replace('_浓度', ''), value=0.0) for dom in dom_cols}

# ==========================================
# 4. 预测逻辑
# ==========================================
if st.button("计算预测值"):
    # 特征预处理
    input_dict = {}
    
    # 对数转换 (必须与训练脚本对齐)
    input_dict['Log_specific surface area m2/g'] = np.log1p(raw_ssa)
    input_dict['Log_molecular weight'] = np.log1p(raw_mw)
    input_dict['Log_adsorption time min'] = np.log1p(raw_time)
    input_dict['Log_C0_to_Dose_Ratio'] = np.log1p(raw_c0 / (raw_dose + 1e-5))
    
    # 线性特征
    if 'pH' in X_cols: input_dict['pH'] = raw_ph
    if 'Ionic strength' in X_cols: input_dict['Ionic strength'] = raw_is
    
    # 类别特征
    for fg, val in user_fgs.items(): input_dict[fg] = float(val)
    for dom, val in user_doms.items(): input_dict[dom] = float(val)
    
    # 构建最终 DataFrame
    final_input = pd.DataFrame([input_dict]).reindex(columns=X_cols, fill_value=0.0)
    
    # 获取最佳模型 (TabM)
    best_name = 'True TabM' if 'True TabM' in models else list(models.keys())[0]
    
    try:
        pred_log = models[best_name].predict(final_input)[0]
        prediction = np.expm1(pred_log)
        
        st.markdown("---")
        st.metric(label=f"预测吸附量 Qm (基于 {best_name})", value=f"{prediction:.2f} mg/g")
        
        # 其他模型对比
        with st.expander("查看其他模型预测参考"):
            for name, model in models.items():
                if name != best_name:
                    p_log = model.predict(final_input)[0]
                    st.write(f"{name}: {np.expm1(p_log):.2f} mg/g")
                    
    except Exception as e:
        st.error(f"预测过程出错: {e}")
