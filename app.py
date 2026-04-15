import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 深度学习架构 (含 Scikit-learn 校验绕过机制)
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
    def fit(self, X, y, sample_weight=None): return self
    def __sklearn_is_fitted__(self): return True
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
    def fit(self, X, y, sample_weight=None): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: 
            m.to(device)
            m.eval()
        with torch.no_grad():
            preds = torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()
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
    def fit(self, X, y, sample_weight=None): return self
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device)
            self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# 命名空间注入
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 缓存加载与中位数提取引擎
# ==========================================
@st.cache_resource
def load_model_pack():
    # 核心修改：仅需将此处挂载点更改为 v6 即可
    return joblib.load('model_artifacts_v8.pkl')

data_pack = load_model_pack()
models = data_pack['models']
X_cols = data_pack['X'].columns.tolist()

X_train_df = data_pack['X']
X_medians = X_train_df.median(numeric_only=True).to_dict()

def get_median(col_name, default_val=0.0):
    return float(X_medians.get(col_name, default_val))

fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_cols = [c for c in X_cols if c.startswith('DOM_')]

log_handled = [
    'Log_specific surface area m2/g', 
    'Log_molecular weight', 
    'Log_adsorption time min', 
    'Log_C0_to_Dose_Ratio'
]

remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_cols and c not in log_handled]

env_cols = []
mat_cols = []

for col in remaining_cols:
    col_lower = col.lower()
    if any(k in col_lower for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        mat_cols.append(col)

# ==========================================
# 3. 界面设计
# ==========================================
st.set_page_config(page_title="Qm Predictor", layout="wide")

if 'theme_choice' not in st.session_state:
    st.session_state.theme_choice = '默认极简 (Light)'

def apply_custom_theme(theme_name):
    if theme_name == '暗夜深邃 (Dark)':
        css = """
        <style>
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #2b2b2b; }
            h1, h2, h3, h4, h5, h6, p, label, span { color: #FFFFFF !important; }
            .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
            .stTabs [data-baseweb="tab"] { color: #FFFFFF; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    elif theme_name == '柔和护眼 (Warm)':
        css = """
        <style>
            .stApp { background-color: #FAEDDF; color: #4A3A2C; }
            [data-testid="stSidebar"] { background-color: #F0DECB; }
            h1, h2, h3, h4, h5, h6, p, label, span { color: #4A3A2C !important; }
            .stTabs [data-baseweb="tab-list"] { background-color: transparent; }
            .stTabs [data-baseweb="tab"] { color: #4A3A2C; }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

st.title("目标污染物吸附性能预测系统")
st.markdown("基于机器学习框架进行理论预测，支持高维特征映射分析。")

with st.sidebar:
    st.subheader("界面设置")
    selected_theme = st.radio(
        "选择主题风格：",
        ('默认极简 (Light)', '暗夜深邃 (Dark)', '柔和护眼 (Warm)'),
        index=0
    )
    apply_custom_theme(selected_theme)
    
    st.markdown("---")
    st.subheader("预测引擎设置")
    best_name = max(data_pack['results'], key=lambda k: data_pack['results'][k]['OOF R2'])
    selected_model = st.selectbox("选择主预测模型", list(models.keys()), index=list(models.keys()).index(best_name))
    st.markdown("系统将以该模型为主输出，并在下方附带其他模型的计算结果以供横向对比。")

user_inputs = {}

tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

# ----------------- 界面 1: 环境与操作条件 -----------------
with tab_env:
    st.subheader("热力学与动力学操作参数")
    col1_e, col2_e = st.columns(2)
    with col1_e:
        if 'Log_C0_to_Dose_Ratio' in X_cols:
            c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.4f", step=0.0001)
            dose = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, format="%.4f", step=0.0001)
            user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0 / (dose + 1e-5))
    with col2_e:
        if 'Log_adsorption time min' in X_cols:
            default_time = np.expm1(get_median('Log_adsorption time min', np.log1p(120.0)))
            val = st.number_input("吸附时间 (min)", value=float(default_time), format="%.4f", step=0.0001)
            user_inputs['Log_adsorption time min'] = np.log1p(val)

    if env_cols:
        st.markdown("---")
        st.subheader("溶液化学环境")
        for col in env_cols:
            user_inputs[col] = st.number_input(f"{col}", value=get_median(col, 0.0), format="%.4f", step=0.0001)

# ----------------- 界面 2: 材料理化特性 -----------------
with tab_mat:
    st.subheader("形貌与高分子特性")
    col1_m, col2_m = st.columns(2)
    with col1_m:
        if 'Log_specific surface area m2/g' in X_cols:
            default_ssa = np.expm1(get_median('Log_specific surface area m2/g', np.log1p(150.0)))
            val = st.number_input("比表面积 (m²/g)", value=float(default_ssa), format="%.4f", step=0.0001)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(val)
    with col2_m:
        if 'Log_molecular weight' in X_cols:
            default_mw = np.expm1(get_median('Log_molecular weight', np.log1p(300.0)))
            val = st.number_input("分子量 (kDa)", value=float(default_mw), format="%.4f", step=0.0001)
            user_inputs['Log_molecular weight'] = np.log1p(val)
            
    if mat_cols:
        st.markdown("---")
        st.subheader("其他物理/化学属性")
        for col in mat_cols:
            user_inputs[col] = st.number_input(f"{col}", value=get_median(col, 0.0), format="%.4f", step=0.0001)

    if fg_cols:
        st.markdown("---")
        st.subheader("表面化学性质 (存在勾选，缺失不勾选)")
        fg_layout_cols = st.columns(3)
        for i, fg in enumerate(fg_cols):
            with fg_layout_cols[i % 3]:
                user_inputs[fg] = float(st.checkbox(fg.replace('FG_', '')))

# ----------------- 界面 3: DOM 干扰 -----------------
with tab_dom:
    st.subheader("溶解性有机质竞争干扰评估")
    if dom_cols:
        for dom in dom_cols:
            user_inputs[dom] = st.number_input(f"{dom.replace('DOM_', '').replace('_浓度', '')} 浓度 (mg/L)", value=0.0, format="%.4f", step=0.0001)
    else:
        st.write("当前模型训练空间未包含 DOM 特征。")

# ==========================================
# 4. 模型推理与参考展示
# ==========================================
st.markdown("---")
if st.button("开始预测", use_container_width=True):
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    fill_dict = {c: 0.0 if (c.startswith('FG_') or c.startswith('DOM_')) else get_median(c, 0.0) for c in X_cols}
    final_df = final_df.fillna(value=fill_dict)
    
    try:
        main_model_pipeline = models[selected_model]
        pred_log = main_model_pipeline.predict(final_df)[0]
        main_prediction = np.expm1(pred_log)
        
        st.success("计算完成")
        st.metric(label=f"主引擎 [{selected_model}] 预测吸附量 Qm (mg/g)", value=f"{main_prediction:.4f}")
        
        st.markdown("#### 其他模型评估参考")
        other_models = [m for m in models.keys() if m != selected_model]
        
        if other_models:
            cols = st.columns(3)
            for i, model_name in enumerate(other_models):
                try:
                    ref_pipeline = models[model_name]
                    ref_pred_log = ref_pipeline.predict(final_df)[0]
                    ref_prediction = np.expm1(ref_pred_log)
                    with cols[i % 3]:
                        st.info(f"✅ **{model_name}**\n\n**{ref_prediction:.4f}** mg/g")
                except Exception:
                    with cols[i % 3]:
                        st.warning(f"⚠️ **{model_name}**\n\n计算失败")
                        
    except Exception as e:
        st.error(f"推理引擎错误: {e}")
