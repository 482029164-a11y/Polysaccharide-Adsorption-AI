import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 深度学习架构 (必须与训练脚本完全对齐)
# ==========================================

class StandardDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1)),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x): return self.network(x)

class PyTorchStandardRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max
        self.T_0 = T_0; self.T_mult = T_mult

    def fit(self, X, y): return self

    def predict(self, X):
        device = torch.device('cpu') 
        if hasattr(self, 'model_'):
            self.model_.to(device)
            self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def __init__(self, k_ensembles=8, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.k_ensembles = k_ensembles; self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max; self.T_0 = T_0; self.T_mult = T_mult

    def fit(self, X, y): return self

    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: 
            m.to(device)
            m.eval()
        with torch.no_grad():
            preds = torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy()
        return np.clip(preds, 0.0, 6.5)

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
    _estimator_type = "regressor"
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.epochs = epochs; self.batch_size = batch_size
        self.lr_min = lr_min; self.lr_max = lr_max
        self.T_0 = T_0; self.T_mult = T_mult

    def fit(self, X, y): return self

    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device)
            self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy()
        return np.clip(preds, 0.0, 6.5)

# ==========================================
# 2. 命名空间注入与模型加载
# ==========================================
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

@st.cache_resource
def load_model_pack():
    return joblib.load('model_artifacts_v3.pkl')

data_pack = load_model_pack()
models = data_pack['models']
X_cols = data_pack['X'].columns.tolist()

# ==========================================
# 3. 界面布局 (移除图标，pH手动填充)
# ==========================================
st.set_page_config(page_title="Qm Predictor", layout="wide")
st.title("多糖基材料吸附量 (Qm) 智能预测系统")

with st.sidebar:
    st.header("预测引擎设置")
    best_name = max(data_pack['results'], key=lambda k: data_pack['results'][k]['OOF R2'])
    selected_model = st.selectbox("选择预测模型", list(models.keys()), index=list(models.keys()).index(best_name))

# 自动分类特征
log_mapping = {
    'Log_specific surface area m2/g': '比表面积 (m²/g)',
    'Log_molecular weight': '分子量 (kDa)',
    'Log_adsorption time min': '吸附时间 (min)',
    'Log_C0_to_Dose_Ratio': 'C0/Dose 比值'
}
fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
# 其他特征（如 pH, 离子强度等）
other_cols = [c for c in X_cols if c not in log_mapping and c not in fg_cols and c not in dom_cols]

col1, col2 = st.columns(2)

user_inputs = {}

with col1:
    st.subheader("基础物理化学参数")
    # 1. 处理对数特征的原始输入
    if 'Log_specific surface area m2/g' in X_cols:
        val = st.number_input("比表面积 Specific Surface Area (m²/g)", value=150.0)
        user_inputs['Log_specific surface area m2/g'] = np.log1p(val)
    
    if 'Log_molecular weight' in X_cols:
        val = st.number_input("分子量 Molecular Weight (kDa)", value=300.0)
        user_inputs['Log_molecular weight'] = np.log1p(val)

    if 'Log_adsorption time min' in X_cols:
        val = st.number_input("吸附时间 Adsorption Time (min)", value=120.0)
        user_inputs['Log_adsorption time min'] = np.log1p(val)

    # 2. 处理 pH 和 离子强度等线性特征 (自动识别)
    for col in other_cols:
        if col == 'Log_C0_to_Dose_Ratio': continue 
        user_inputs[col] = st.number_input(f"{col}", value=0.0, format="%.4f" if "strength" in col.lower() else "%.2f")

with col2:
    st.subheader("实验条件与官能团")
    # 3. 特殊处理 C0/Dose 比值
    if 'Log_C0_to_Dose_Ratio' in X_cols:
        c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0)
        dose = st.number_input("投加量 Dose (mg/ml)", value=1.0)
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0 / (dose + 1e-5))

    # 4. 官能团 (Checkbox)
    if fg_cols:
        st.write("选择存在的官能团:")
        for fg in fg_cols:
            user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

# 5. DOM 
if dom_cols:
    with st.expander("环境因子 (DOM)"):
        for dom in dom_cols:
            user_inputs[dom] = st.number_input(dom.replace('DOM_', ''), value=0.0)

# ==========================================
# 4. 预测执行
# ==========================================
if st.button("开始预测"):
    # 构建 DataFrame 并严格对齐训练集的列顺序
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols, fill_value=0.0)
    
    # 获取选中的模型 Pipeline
    model_pipeline = models[selected_model]
    
    try:
        # 使用 Pipeline 预测，确保包含 StandardScaler 步骤，解决预测为 0 的问题
        pred_log = model_pipeline.predict(final_df)[0]
        prediction = np.expm1(pred_log)
        
        st.success(f"预测完成")
        st.metric("预测吸附量 Qm (mg/g)", f"{prediction:.4f}")
        
    except Exception as e:
        st.error(f"预测出错: {e}")
