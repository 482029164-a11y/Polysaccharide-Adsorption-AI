import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
import os
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 核心架构类定义 (全量补全以支持 v16 内核加载)
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
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    def predict(self, X):
        device = torch.device('cpu')
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        for m in self.models_: m.to(device); m.eval()
        with torch.no_grad():
            preds = torch.cat([m(X_t) for m in self.models_], dim=1).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

class TrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, k_ensembles=32, dropout=0.1):
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
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# 🚀 强制注入命名空间，确保 joblib 反序列化成功
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 内核加载引擎
# ==========================================
@st.cache_resource
def load_v16_kernel():
    path = 'model_artifacts_v16.pkl'
    if not os.path.exists(path):
        st.error(f"找不到内核文件: {path}")
        st.stop()
    try:
        pack = joblib.load(path)
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        # 默认调用 v16 表现最优专家
        model = pack['models']['True TabM']
        return X_cols, X_medians, model
    except Exception as e:
        st.error(f"内核加载失败: {e}")
        st.stop()

X_cols, X_medians, model_engine = load_v16_kernel()

# ==========================================
# 3. 页面布局与输入 (严格固定特征排版)
# ==========================================
st.set_page_config(page_title="吸附性能预测 v16", layout="wide")
st.title("目标污染物吸附性能预测系统 (v16)")
st.info("系统状态：True TabM 专家引擎已就绪 | 物理门控机制已自动激活")

user_inputs = {}
# 分类列表：确保覆盖所有特征
env_keys = ['ph', 'temp', 'speed', 'rpm', 'time', 'concentration', 'ratio', 'dose']
mat_keys = ['surface', 'weight', 'pore', 'fg_', 'ash', 'carbon', 'oxygen', 'nitrogen']
dom_keys = ['dom_', 'ha', 'fa', 'ca']

tab_env, tab_mat, tab_dom = st.tabs(["反应环境与操作条件", "材料理化与结构特性", "共存水体基质 (DOM)"])

with tab_env:
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        c0_raw = st.number_input("初始浓度 C0 (mg/L)", value=50.0, format="%.2f")
        dose_raw = st.number_input("投加量 Dose (mg/ml)", value=1.0, format="%.2f")
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_raw / (dose_raw + 1e-7))
    with col_e2:
        time_raw = st.number_input("吸附接触时间 (min)", value=120.0, format="%.2f")
        user_inputs['Log_adsorption time min'] = np.log1p(time_raw)
    
    st.markdown("---")
    # 动态渲染其余环境特征
    for col in X_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in env_keys) and col not in user_inputs:
            user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)), format="%.4f")

with tab_mat:
    # 1. 结构化特征 (对数处理)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        if 'Log_specific surface area m2/g' in X_cols:
            ssa_raw = st.number_input("比表面积 SSA (m2/g)", value=float(np.expm1(X_medians.get('Log_specific surface area m2/g', 5.0))))
            user_inputs['Log_specific surface area m2/g'] = np.log1p(ssa_raw)
    with col_m2:
        if 'Log_molecular weight' in X_cols:
            mw_raw = st.number_input("分子量 (kDa)", value=float(np.expm1(X_medians.get('Log_molecular weight', 5.0))))
            user_inputs['Log_molecular weight'] = np.log1p(mw_raw)
            
    st.markdown("---")
    # 2. 官能团与元素组成 (FG_Phosphate 强制包含在此)
    st.subheader("表面官能团与元素属性")
    col_fg1, col_fg2 = st.columns(2)
    fg_list = [c for c in X_cols if c.startswith('FG_')]
    for i, col in enumerate(fg_list):
        with (col_fg1 if i % 2 == 0 else col_fg2):
            user_inputs[col] = float(st.checkbox(col.replace('FG_', ''), value=False))
            
    # 3. 其他理化指标
    for col in X_cols:
        col_lower = col.lower()
        if any(k in col_lower for k in mat_keys) and col not in user_inputs and not col.startswith('FG_'):
            user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)), format="%.4f")

with tab_dom:
    st.subheader("竞争抑制因子输入")
    dom_list = [c for c in X_cols if c.startswith('DOM_')]
    for col in dom_list:
        user_inputs[col] = st.number_input(f"{col.replace('DOM_', '')} 浓度 (mg/L)", value=0.0, format="%.4f")

# ==========================================
# 4. 推理核心：物理门控自动注入
# ==========================================
st.markdown("---")
if st.button("开始运行 v16 物理引导预测", use_container_width=True):
    # 核心物理逻辑：自动判定 Physical_Gate_Inhibition
    ha_val = user_inputs.get('DOM_HA', 0.0)
    # 判定准则：极低浓度 (<10) 且有腐殖酸存在时，激活抑制门控
    gate_signal = 1.0 if (c0_raw < 10.0 and ha_val > 0) else 0.0
    user_inputs['Physical_Gate_Inhibition'] = gate_signal
    
    # 构建 DataFrame 并对齐 X_cols 顺序 (无遗漏补齐)
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    
    # 填补可能存在的 NaN (使用训练集中位数)
    for col in X_cols:
        if pd.isna(final_df[col][0]):
            final_df[col] = X_medians.get(col, 0.0)
            
    try:
        # 预测并逆变换
        pred_log = model_engine.predict(final_df)[0]
        final_qm = np.expm1(pred_log)
        
        # 结果展示
        if gate_signal > 0.5:
            st.warning("🎯 **物理门控激活**：检测到极低浓度强竞争环境。模型已启动局部抑制流形进行数值纠偏。")
        else:
            st.success("✅ **物理状态正常**：当前处于全局热力学推理模式。")
            
        st.metric(label="理论平衡吸附量 Qm (mg/g)", value=f"{final_qm:.4f}")
        
    except Exception as e:
        st.error(f"推理引擎异常: {e}")

# ==========================================
# 5. 系统侧边栏
# ==========================================
st.sidebar.markdown("### 系统规格")
st.sidebar.write("内核版本: v16.0-Final")
st.sidebar.caption("锁定特征总数: " + str(len(X_cols)))
st.sidebar.caption("主专家模型: True TabM (32 Heads)")
st.sidebar.markdown("---")
st.sidebar.caption("注意：Physical_Gate_Inhibition 是由系统根据 C0 和 DOM_HA 自动计算的掩码特征，无需手动输入。")
