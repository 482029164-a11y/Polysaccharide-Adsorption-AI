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
# 3. 界面布局 (三界面结构)
# ==========================================
st.set_page_config(page_title="Qm Predictor", layout="centered")
st.title("多糖基材料吸附量预测系统")

# 侧边栏：选择主预测引擎
with st.sidebar:
    st.header("预测引擎设置")
    best_name = max(data_pack['results'], key=lambda k: data_pack['results'][k]['OOF R2'])
    selected_model = st.selectbox("选择主预测模型", list(models.keys()), index=list(models.keys()).index(best_name))
    st.markdown("---")
    st.markdown("系统将以该模型作为核心输出，同时在下方交叉验证区域提供其他模型的计算结果以供参考。")

# 自动分类特征
log_mapping = {
    'Log_specific surface area m2/g': '比表面积 (m²/g)',
    'Log_molecular weight': '分子量 (kDa)',
    'Log_adsorption time min': '吸附时间 (min)',
    'Log_C0_to_Dose_Ratio': 'C0/Dose 比值'
}
fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_cols = [c for c in X_cols if c.startswith('DOM_')]
other_cols = [c for c in X_cols if c not in log_mapping and c not in fg_cols and c not in dom_cols]

# 初始化用户输入字典
user_inputs = {}

# 使用 st.tabs 创建三个界面分区
tab_env, tab_mat, tab_dom = st.tabs(["反应环境条件", "材料固有属性", "DOM 浓度特征"])

# 界面1：反应环境条件
with tab_env:
    st.subheader("物理环境设置")
    
    # 吸附时间
    if 'Log_adsorption time min' in X_cols:
        val = st.number_input("吸附时间 (min)", value=120.0, step=10.0)
        user_inputs['Log_adsorption time min'] = np.log1p(val)

    # 初始浓度与投加量
    if 'Log_C0_to_Dose_Ratio' in X_cols:
        c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0, step=5.0)
        dose = st.number_input("投加量 Dose (mg/ml)", value=1.0, step=0.1)
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0 / (dose + 1e-5))
        
    # 其他环境特征 (pH, 离子强度等)
    for col in other_cols:
        user_inputs[col] = st.number_input(f"{col}", value=0.0, format="%.4f" if "strength" in col.lower() else "%.2f")

# 界面2：材料固有属性
with tab_mat:
    st.subheader("物理与结构参数")
    
    if 'Log_specific surface area m2/g' in X_cols:
        val = st.number_input("比表面积 (m²/g)", value=150.0, step=10.0)
        user_inputs['Log_specific surface area m2/g'] = np.log1p(val)
        
    if 'Log_molecular weight' in X_cols:
        val = st.number_input("分子量 (kDa)", value=300.0, step=50.0)
        user_inputs['Log_molecular weight'] = np.log1p(val)

    st.subheader("表面官能团 (勾选即存在)")
    if fg_cols:
        # 使用多列显示官能团，节约纵向空间
        fg_layout_cols = st.columns(3)
        for i, fg in enumerate(fg_cols):
            with fg_layout_cols[i % 3]:
                user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

# 界面3：DOM浓度特征
with tab_dom:
    st.subheader("溶解性有机质 (DOM) 共存干扰")
    if dom_cols:
        for dom in dom_cols:
            user_inputs[dom] = st.number_input(f"{dom.replace('DOM_', '').replace('_浓度', '')} 浓度 (mg/L)", value=0.0, step=1.0)
    else:
        st.info("模型训练数据中未检测到有效的 DOM 浓度特征列。")

# ==========================================
# 4. 预测执行与多模型对比
# ==========================================
st.markdown("---")
if st.button("开始预测", use_container_width=True):
    # 构建 DataFrame 并严格对齐训练集的列顺序
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols, fill_value=0.0)
    
    try:
        # 1. 核心模型预测
        model_pipeline = models[selected_model]
        pred_log = model_pipeline.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        
        st.success("预测计算已完成")
        
        # 核心结果展示
        st.metric(label=f"主引擎 [{selected_model}] 预测吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        
        # 2. 其他模型参考对比
        st.markdown("#### 其他模型评估参考")
        
        # 为了美观，使用 columns 平铺其他模型的结果
        other_models = [m for m in models.keys() if m != selected_model]
        
        if other_models:
            # 动态生成列布局，每行放 3 个模型
            cols = st.columns(3)
            for i, model_name in enumerate(other_models):
                try:
                    ref_pipeline = models[model_name]
                    ref_pred_log = ref_pipeline.predict(final_df)[0]
                    ref_prediction = np.expm1(ref_pred_log)
                    
                    with cols[i % 3]:
                        st.info(f"**{model_name}**\n\n{ref_prediction:.4f} mg/g")
                except Exception:
                    with cols[i % 3]:
                        st.warning(f"**{model_name}**\n\n计算失败")
        
    except Exception as e:
        st.error(f"预测过程出错: {e}")
