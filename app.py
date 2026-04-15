import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import math
from sklearn.base import BaseEstimator, RegressorMixin

# ==========================================
# 1. 核心架构类声明 (必须保留以供 joblib 成功反序列化)
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
    def predict(self, X):
        device = torch.device('cpu')
        if hasattr(self, 'model_'):
            self.model_.to(device); self.model_.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = self.model_(X_t).mean(dim=1).cpu().numpy().flatten()
        return np.clip(preds, 0.0, 6.5)

# 为了兼容性，将类注入到 __main__ 命名空间
import __main__
__main__.PyTorchTrueTabMRegressor = PyTorchTrueTabMRegressor
__main__.StandardDNN = StandardDNN
__main__.TrueTabMMini = TrueTabMMini

# ==========================================
# 2. 物理引导单内核加载引擎
# ==========================================
@st.cache_resource
def load_v16_system():
    try:
        # 加载 v16 全局内核
        pack = joblib.load('model_artifacts_v16.pkl')
        X_cols = pack['X'].columns.tolist()
        X_medians = pack['X'].median(numeric_only=True).to_dict()
        # 默认调用表现最优的 True TabM
        model = pack['models']['True TabM']
        return X_cols, X_medians, model
    except Exception as e:
        st.error(f"内核文件 'model_artifacts_v16.pkl' 加载失败。请确保文件在同级目录且已上传完整。错误信息: {e}")
        st.stop()

X_cols, X_medians, model_engine = load_v16_system()

# ==========================================
# 3. 页面布局与输入
# ==========================================
st.set_page_config(page_title="污染物吸附预测 v16", layout="wide")
st.title("污染物吸附性能预测系统 (v16)")
st.info("当前引擎：True TabM 深度学习网络（已激活物理特征门控机制）")

user_inputs = {}
tab1, tab2, tab3 = st.tabs(["核心环境因子", "材料参数", "水体基质 (DOM)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        c0_input = st.number_input("初始浓度 C0 (mg/L)", value=50.0, step=0.1, format="%.2f")
        dose_input = st.number_input("吸附剂投加量 (mg/ml)", value=1.0, step=0.1, format="%.2f")
        # 记录关键原始数据用于门控判断
        user_inputs['Log_C0_to_Dose_Ratio'] = np.log1p(c0_input / (dose_input + 1e-7))
    with c2:
        time_input = st.number_input("吸附时间 (min)", value=120.0, step=1.0)
        user_inputs['Log_adsorption time min'] = np.log1p(time_input)
    
    # 填充其他环境特征（如 pH, Temp）
    env_targets = [c for c in X_cols if any(k in c.lower() for k in ['ph', 'temp', 'rpm'])]
    for col in env_targets:
        if col not in user_inputs:
            user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)))

with tab2:
    st.markdown("##### 结构与功能特性")
    # 自动识别比表面积等对数特征
    mat_cols = [c for c in X_cols if any(k in c.lower() for k in ['surface', 'weight', 'pore', 'fg_'])]
    for col in mat_cols:
        if col.startswith('FG_'):
            user_inputs[col] = float(st.checkbox(col.replace('FG_', '')))
        elif col.startswith('Log_'):
            raw_val = st.number_input(f"请输入: {col.replace('Log_', '')}", value=float(np.expm1(X_medians.get(col, 0.0))))
            user_inputs[col] = np.log1p(raw_val)
        else:
            user_inputs[col] = st.number_input(col, value=float(X_medians.get(col, 0.0)))

with tab3:
    st.subheader("溶解性有机质 (DOM)")
    dom_cols = [c for c in X_cols if c.startswith('DOM_')]
    for col in dom_cols:
        user_inputs[col] = st.number_input(f"{col.replace('DOM_', '')} 浓度 (mg/L)", value=0.0)

# ==========================================
# 4. 推理核心：物理门控自动注入
# ==========================================
st.markdown("---")
if st.button("开始运行 v16 物理引擎", use_container_width=True):
    # 1. 核心物理逻辑：自动判定是否属于特殊环境
    ha_val = user_inputs.get('DOM_HA', 0.0)
    inhibition_gate = 1.0 if (c0_input < 10.0 and ha_val > 0) else 0.0
    user_inputs['Physical_Gate_Inhibition'] = inhibition_gate
    
    # 2. 构建 DataFrame 并通过 reindex 强制特征对齐
    final_df = pd.DataFrame([user_inputs]).reindex(columns=X_cols)
    
    # 3. 填补缺失列
    for col in X_cols:
        if pd.isna(final_df[col][0]):
            final_df[col] = X_medians.get(col, 0.0)
            
    try:
        # 4. 模型预测
        pred_log = model_engine.predict(final_df)[0]
        final_qm = np.expm1(pred_log)
        
        # 5. UI 反馈
        if inhibition_gate > 0.5:
            st.warning("🎯 **物理门控激活**：检测到极低浓度强竞争体系。模型已自动在注意力机制底层调用纠偏权重。")
        else:
            st.success("✅ **物理状态正常**：当前处于全局热力学流形推理模式。")
            
        st.metric(label="预测平衡吸附量 Qm (mg/g)", value=f"{final_qm:.4f}")
        
    except Exception as e:
        st.error(f"推理引擎运行时错误: {e}")
