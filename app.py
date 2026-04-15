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
    def fit(self, X, y): return self
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
    def fit(self, X, y): return self
    def __sklearn_is_fitted__(self): return True
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
    def __sklearn_is_fitted__(self): return True
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device)
            self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy()
        return np.clip(preds, 0.0, 6.5)

# 命名空间注入，防止 joblib 找不到类
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
    return joblib.load('model_artifacts_v3.pkl')

data_pack = load_model_pack()
models = data_pack['models']
X_cols = data_pack['X'].columns.tolist()

# --- 提取训练集数据并计算中位数 ---
X_train_df = data_pack['X']
X_medians = X_train_df.median(numeric_only=True).to_dict()

def get_median(col_name, default_val=0.0):
    return float(X_medians.get(col_name, default_val))

# -- 基于材料学视角的特征分类 --
fg_cols = [c for c in X_cols if c.startswith('FG_')]
dom_cols = [c for c in X_cols if c.startswith('DOM_')]

# 已手动处理对数转换的核心变量
log_handled = [
    'Log_specific surface area m2/g', 
    'Log_molecular weight', 
    'Log_adsorption time min', 
    'Log_C0_to_Dose_Ratio'
]

# 剩余未分类特征归属
remaining_cols = [c for c in X_cols if c not in fg_cols and c not in dom_cols and c not in log_handled]

env_cols = []
mat_cols = []

for col in remaining_cols:
    col_lower = col.lower()
    # 仅对模型数据中客观存在的特征进行分类
    if any(k in col_lower for k in ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration']):
        env_cols.append(col)
    else:
        # 归属为材料属性 (如 porosity, pore size, zeta potential等)
        mat_cols.append(col)

# ==========================================
# 3. 界面设计 (三标签页结构)
# ==========================================
st.set_page_config(page_title="Qm Predictor", layout="wide")

# --- 动态主题控制逻辑 ---
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

st.title("多糖基材料重金属吸附性能预测系统")
st.markdown("基于机器学习框架进行理论预测，提供多模型集成均值（Ensemble）以供参考。")

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
    st.markdown("系统将自动集成所有可用模型（Ensemble），采用 **IQR 统计算法** 剔除外推失真的离群模型，并将剩余稳健模型的**数学平均值**作为最终输出。")

user_inputs = {}

# 核心三板块标签页
tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

# ----------------- 界面 1: 环境与操作条件 -----------------
with tab_env:
    st.subheader("热力学与动力学操作参数")
    
    col1_e, col2_e = st.columns(2)
    with col1_e:
        if 'Log_C0_to_Dose_Ratio' in X_cols:
            c0 = st.number_input("初始浓度 C0 (mg/L)", value=50.0, step=5.0)
            dose = st.number_input("吸附剂投加量 Dose (mg/ml)", value=1.0, step=0.1)
            user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0 / (dose + 1e-5))
    
    with col2_e:
        if 'Log_adsorption time min' in X_cols:
            # 还原对数值并用作默认显示
            default_time = np.expm1(get_median('Log_adsorption time min', np.log1p(120.0)))
            val = st.number_input("吸附时间 (min)", value=float(default_time), step=10.0)
            user_inputs['Log_adsorption time min'] = np.log1p(val)

    if env_cols:
        st.markdown("---")
        st.subheader("溶液化学环境")
        for col in env_cols:
            format_str = "%.2f"
            user_inputs[col] = st.number_input(f"{col}", value=get_median(col, 0.0), format=format_str)

# ----------------- 界面 2: 材料理化特性 -----------------
with tab_mat:
    st.subheader("形貌与高分子特性")
    
    col1_m, col2_m = st.columns(2)
    with col1_m:
        if 'Log_specific surface area m2/g' in X_cols:
            default_ssa = np.expm1(get_median('Log_specific surface area m2/g', np.log1p(150.0)))
            val = st.number_input("比表面积 (m²/g)", value=float(default_ssa), step=10.0)
            user_inputs['Log_specific surface area m2/g'] = np.log1p(val)
    with col2_m:
        if 'Log_molecular weight' in X_cols:
            default_mw = np.expm1(get_median('Log_molecular weight', np.log1p(300.0)))
            val = st.number_input("分子量 (kDa)", value=float(default_mw), step=50.0)
            user_inputs['Log_molecular weight'] = np.log1p(val)
            
    # 动态渲染其余材料属性
    if mat_cols:
        st.markdown("---")
        st.subheader("其他物理/化学属性")
        for col in mat_cols:
            user_inputs[col] = st.number_input(f"{col}", value=get_median(col, 0.0))

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
            user_inputs[dom] = st.number_input(f"{dom.replace('DOM_', '').replace('_浓度', '')} 浓度 (mg/L)", value=0.0, step=1.0)
    else:
        st.write("当前模型训练空间未包含 DOM 特征。")

# ==========================================
# 4. 模型推理与稳健集成(Robust Ensemble)展示
# ==========================================
st.markdown("---")
if st.button("开始预测", use_container_width=True):
    # 构建 DataFrame
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    
    # --- 智能缺失值兜底填充 ---
    # 官能团和DOM填0，连续特征填中位数
    fill_dict = {c: 0.0 if (c.startswith('FG_') or c.startswith('DOM_')) else get_median(c, 0.0) for c in X_cols}
    final_df = final_df.fillna(value=fill_dict)
    
    try:
        all_preds = {}
        # 遍历所有模型进行预测
        for name, model_pipeline in models.items():
            try:
                pred_log = model_pipeline.predict(final_df)[0]
                all_preds[name] = np.expm1(pred_log)
            except Exception as e:
                pass # 忽略推理失败的个别模型

        if not all_preds:
            st.error("所有模型推理均失败，请检查数据输入或环境配置。")
        else:
            # ---------------------------------------------------------
            # 💡 核心修改：基于 IQR 的异常模型剔除算法
            # ---------------------------------------------------------
            preds_array = np.array(list(all_preds.values()))
            q1 = np.percentile(preds_array, 25)
            q3 = np.percentile(preds_array, 75)
            iqr = q3 - q1
            
            # 将 IQR 乘数降至 1.0，最低容忍误差降至 5%
            tolerance = max(0.5 * iqr, 0.02 * np.median(preds_array))
            lower_bound = q1 - tolerance
            upper_bound = q3 + tolerance
            
            normal_preds = {}
            outlier_preds = {}
            
            for m_name, p_val in all_preds.items():
                if lower_bound <= p_val <= upper_bound:
                    normal_preds[m_name] = p_val
                else:
                    outlier_preds[m_name] = p_val
                    
            # 极端情况兜底：如果检测算法意外剔除了所有模型，则回退为全量平均
            if not normal_preds:
                normal_preds = all_preds
                outlier_preds = {}

            # 计算稳健集成均值（仅针对未被剔除的模型）
            ensemble_prediction = np.mean(list(normal_preds.values()))
            
            st.success("计算完成")
            st.metric(label="综合稳健预测吸附量 Qm (剔除异常模型后的均值, mg/g)", value=f"{ensemble_prediction:.4f}")
            
            if outlier_preds:
                st.warning(f"系统检测到 {len(outlier_preds)} 个模型的预测值发生严重偏离，已在底层均值计算中将其剔除。")
            
            # 独立模型拆解参考展示
            st.markdown("#### 各独立模型预测详情")
            cols = st.columns(3)
            
            for i, (model_name, pred_val) in enumerate(all_preds.items()):
                with cols[i % 3]:
                    if model_name in outlier_preds:
                        # 异常模型：标红警告，注明被剔除
                        st.error(f"⚠️ **{model_name}**\n\n**{pred_val:.4f}** mg/g\n\n*(偏差过大，已自动剔除)*")
                    else:
                        # 正常模型：标准绿色提示框
                        st.info(f"✅ **{model_name}**\n\n**{pred_val:.4f}** mg/g")
                    
    except Exception as e:
        st.error(f"推理引擎整体错误: {e}")
