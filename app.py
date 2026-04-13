import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
import sys
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 深度学习架构 (严格对齐训练脚本)
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
        # 确保输入是 float32 且在 CPU 上
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
    # 强制加载到 CPU 避免 CUDA 错误
    return joblib.load('model_artifacts_v3.pkl')

try:
    data_pack = load_model_pack()
    models = data_pack['models']
    X_cols = data_pack['X'].columns.tolist()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# ==========================================
# 3. 界面布局 (去除所有图标)
# ==========================================
st.set_page_config(page_title="Qm Adsorption Predictor", layout="wide")

st.title("多糖基材料吸附量 (Qm) 智能预测系统")
st.markdown("说明：输入实验参数后，点击下方预测按钮。如果某项指标在您的原始实验中不存在，请保持默认值（通常为0或均值）。")

with st.sidebar:
    st.header("预测引擎设置")
    best_model_name = max(data_pack['results'], key=lambda k: data_pack['results'][k]['OOF R2'])
    selected_model_name = st.selectbox("选择预测模型", list(models.keys()), index=list(models.keys()).index(best_model_name))
    st.info(f"当前推荐最佳模型: {best_model_name}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("材料物理化学性质")
    # 针对可能缺失的特征，检查是否存在于 X_cols 中
    raw_ssa = st.number_input("比表面积 Specific Surface Area (m²/g)", value=150.0)
    raw_mw = st.number_input("分子量 Molecular Weight (kDa)", value=300.0)
    # 修改 1：pH 手动填充
    raw_ph = st.number_input("溶液 pH 值", value=5.5, step=0.1)
    
    # 仅当模型训练时包含该特征，才显示输入框，否则设为隐藏默认值
    if 'Ionic strength' in X_cols:
        raw_is = st.number_input("离子强度 Ionic Strength (mol/L)", value=0.01, format="%.4f")
    else:
        raw_is = 0.01

with col2:
    st.subheader("实验反应条件")
    raw_time = st.number_input("吸附时间 Adsorption Time (min)", value=120.0)
    raw_c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0)
    raw_dose = st.number_input("投加量 Dose (mg/ml)", value=1.0)
    
    st.subheader("官能团信息")
    fg_cols = [c for c in X_cols if c.startswith('FG_')]
    user_fgs = {}
    if fg_cols:
        fg_sub_cols = st.columns(2)
        for i, fg in enumerate(fg_cols):
            with fg_sub_cols[i % 2]:
                user_fgs[fg] = st.checkbox(fg.replace('FG_', ''), value=False)

# 环境因子 (DOM)
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
if dom_cols:
    with st.expander("环境共存离子/DOM 浓度 (mg/L)"):
        dom_sub_cols = st.columns(3)
        user_doms = {}
        for i, dom in enumerate(dom_cols):
            with dom_sub_cols[i % 3]:
                user_doms[dom] = st.number_input(dom.replace('DOM_', ''), value=0.0)
else:
    user_doms = {}

# ==========================================
# 4. 预测逻辑 (修复预测为 0 的问题)
# ==========================================
if st.button("开始预测吸附量 Qm"):
    try:
        # 1. 准备输入字典 (必须与训练时的特征名完全匹配)
        input_dict = {}
        
        # 严格对数转换逻辑
        input_dict['Log_specific surface area m2/g'] = np.log1p(raw_ssa)
        input_dict['Log_molecular weight'] = np.log1p(raw_mw)
        input_dict['Log_adsorption time min'] = np.log1p(raw_time)
        input_dict['Log_C0_to_Dose_Ratio'] = np.log1p(raw_c0 / (raw_dose + 1e-5))
        
        # 只有训练集中有的特征才填入
        if 'pH' in X_cols: 
            input_dict['pH'] = raw_ph
        if 'Ionic strength' in X_cols: 
            input_dict['Ionic strength'] = raw_is
        
        # 官能团与 DOM
        for fg, val in user_fgs.items(): input_dict[fg] = float(val)
        for dom, val in user_doms.items(): input_dict[dom] = float(val)
        
        # 2. DataFrame 对齐 (关键：使用 reindex 并填充训练集缺失的列)
        # 如果原始数据里没有 Ionic strength，reindex 会自动补 0，确保 DNN 输入维度正确
        final_input = pd.DataFrame([input_dict]).reindex(columns=X_cols, fill_value=0.0)
        
        # 3. 推理 (通过 Pipeline 调用，自动包含 StandardScaler)
        model = models[selected_model_name]
        
        # 调试信息：如果预测总是 0，可以暂时取消下方注释查看输入是否异常
        # st.write(final_input) 
        
        # 调用的是 Pipeline 的 predict，它会先调用 scaler.transform() 再调用 model.predict()
        pred_log = model.predict(final_input)[0]
        
        # 指数反转
        prediction = np.expm1(pred_log)
        
        # 4. 结果展示
        if prediction < 0.0001:
            st.warning("预测值接近于 0，请检查输入参数是否在训练集范围内，或尝试切换其他模型（如随机森林）。")
        
        st.success("预测成功")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="预测吸附量 Qm (mg/g)", value=f"{prediction:.4f}")
        with res_col2:
            st.info(f"使用的模型: {selected_model_name}")
            
        with st.expander("查看其他模型预测对比"):
            for name, m in models.items():
                if name != selected_model_name:
                    p_log = m.predict(final_input)[0]
                    st.write(f"**{name}**: {np.expm1(p_log):.4f} mg/g")

    except Exception as e:
        st.error(f"预测计算失败: {str(e)}")
        st.info("建议：请检查 .pkl 文件是否为最新训练版本，且包含所有必要的类定义。")

st.markdown("---")
st.caption("提示：如果 DNN 类模型输出异常，通常是因为输入特征分布与训练集偏差过大，导致神经网络神经元失活。")
