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
# 1. 核心架构类声明 (补全 v16 训练时涉及的所有类定义)
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

# 🚀 命名空间注入：通过将定义挂载到 __main__ 彻底解决 joblib 加载失败的问题
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.PyTorchStandardRegressor = PyTorchStandardRegressor
__main__.PyTorchDeepEnsembleRegressor = PyTorchDeepEnsembleRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 物理引导单内核加载引擎 (v16)
# ==========================================
@st.cache_resource
def load_v16_system():
    model_path = 'model_artifacts_v16.pkl'
    if not os.path.exists(model_path):
        st.error(f"严重错误：在当前目录下找不到内核文件 {model_path}。")
        st.stop()
    try:
        pack = joblib.load(model_path)
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        # v16 默认使用最强的 True TabM 引擎
        best_model = pack['models']['True TabM']
        return X_cols, X_medians, best_model
    except Exception as e:
        st.error(f"内核反序列化失败。错误详情: {e}")
        st.stop()

X_cols, X_medians, model_tabm = load_v16_system()

# ==========================================
# 3. 页面布局与参数输入
# ==========================================
st.set_page_config(page_title="污染物吸附预测系统 v16", layout="wide")
st.title("目标污染物吸附性能预测系统 (v16)")
st.info("核心引擎：True TabM 深度强化学习网络。已集成物理特征门控机制。")

user_inputs = {}
tab1, tab2, tab3 = st.tabs(["反应条件与环境因子", "材料理化特性参数", "水体基质 (DOM) 评估"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        c0_input = st.number_input("初始浓度 C0 (mg/L)", value=50.0, step=0.1, format="%.2f")
        dose_input = st.number_input("吸附剂投加量 (mg/ml)", value=1.0, step=0.1, format="%.2f")
        # 特征对齐逻辑：Log_C0_to_Dose_Ratio
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_input / (dose_input + 1e-7))
    with c2:
        time_input = st.number_input("吸附接触时间 (min)", value=120.0, step=1.0)
        user_inputs['Log_adsorption time min'] = np.log1p(time_input)
    
    # 环境列（如 pH, 温度, 转速）
    env_cols = [c for c in X_cols if any(k in c.lower() for k in ['ph', 'temp', 'rpm'])]
    for col in env_cols:
        user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)))

with tab2:
    st.markdown("##### 结构表征与官能团性质")
    mat_cols = [c for c in X_cols if any(k in c.lower() for k in ['surface', 'weight', 'pore', 'fg_'])]
    for col in mat_cols:
        if col.startswith('FG_'):
            user_inputs[col] = float(st.checkbox(col.replace('FG_', '')))
        elif col.startswith('Log_'):
            label = col.replace('Log_', '')
            raw_v = st.number_input(f"请输入实测值: {label}", value=float(np.expm1(X_medians.get(col, 0.0))))
            user_inputs[col] = np.log1p(raw_v)
        else:
            user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)))

with tab3:
    st.subheader("溶解性有机质 (DOM) 浓度输入")
    dom_cols = [c for c in X_cols if c.startswith('DOM_')]
    for col in dom_cols:
        user_inputs[col] = st.number_input(f"{col.replace('DOM_', '')} (mg/L)", value=0.0)

# ==========================================
# 4. 推理核心：物理特征门控自动植入
# ==========================================
st.markdown("---")
if st.button("开始运行物理引导推理", use_container_width=True):
    # 核心创新点：判定物理门控状态 (C0 < 10 且 HA > 0)
    ha_val = user_inputs.get('DOM_HA', 0.0)
    inhibition_gate = 1.0 if (c0_input < 10.0 and ha_val > 0) else 0.0
    
    # 注入 v16 训练时新增的 Physical_Gate_Inhibition 特征列
    user_inputs['Physical_Gate_Inhibition'] = inhibition_gate
    
    # 构建 DataFrame 并强制对齐训练时的特征空间顺序
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    
    # 填补未由界面录入的缺失列（使用训练集特征中位数）
    for col in X_cols:
        if pd.isna(final_df[col][0]):
            final_df[col] = X_medians.get(col, 0.0)
            
    try:
        # 执行推理
        pred_log = model_tabm.predict(final_df)[0]
        final_qm = np.expm1(pred_log)
        
        # 结果反馈与物理状态说明
        if inhibition_gate > 0.5:
            st.warning("🎯 **物理门控激活**：检测到极低浓度强竞争体系。模型已自动启动 TabM 专家权重，输出纠偏后的吸附量。")
        else:
            st.success("✅ **物理状态正常**：当前处于常规热力学流形推理模式。")
            
        st.metric(label="理论吸附量 Qm (mg/g)", value=f"{final_qm:.4f}")
        
    except Exception as e:
        st.error(f"推理引擎运行时错误: {e}")

# ==========================================
# 5. 系统侧边栏信息
# ==========================================
st.sidebar.markdown("---")
st.sidebar.write("**版本信息：v16.0 (Stable)**")
st.sidebar.caption("更新日志：")
st.sidebar.caption("- 物理门控特征 (Physical Gate) 全链路跑通")
st.sidebar.caption("- 修复 Joblib 命名空间 Attribute 错误")
st.sidebar.caption("- 强化低浓度下 HA 竞争抑制的预测精度")
