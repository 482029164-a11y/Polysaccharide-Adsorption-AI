import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

warnings.filterwarnings('ignore')


# ==========================================
# 🌟 必须同步的深度学习类蓝图 (确保顺利反序列化)
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
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y): pass

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        self.model_.eval()
        with torch.no_grad(): preds = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return np.clip(preds.flatten(), 0.0, 6.5)


class PyTorchDeepEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k_ensembles=8, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y):
        pass

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        for m in self.models_: m.eval()
        with torch.no_grad():
            preds = torch.cat([m(torch.tensor(X, dtype=torch.float32).to(self.device)) for m in self.models_],
                              dim=1).mean(dim=1).cpu().numpy()
        return np.clip(preds, 0.0, 6.5)


class TrueTabMMini(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, k_ensembles=32, dropout=0.1):
        super().__init__()
        self.k_ensembles = k_ensembles
        self.R = nn.Parameter(torch.randn(k_ensembles, 1, input_dim))
        self.shared_bottom = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(),
            nn.Dropout(max(0, dropout - 0.1))
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim // 2, 1) for _ in range(k_ensembles)])

    def forward(self, x):
        batch_size = x.size(0)
        x_rep = x.unsqueeze(0).repeat(self.k_ensembles, 1, 1) * self.R
        out_flat = self.shared_bottom(x_rep.view(-1, x.size(-1)))
        out = out_flat.view(self.k_ensembles, batch_size, -1)
        return torch.cat([self.heads[i](out[i]) for i in range(self.k_ensembles)], dim=1)


class PyTorchTrueTabMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=250, batch_size=32, lr_min=0.0001, lr_max=0.002, T_0=50, T_mult=1.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, X, y): pass

    def predict(self, X):
        if isinstance(X, pd.DataFrame): X = X.values
        self.model_.eval()
        with torch.no_grad(): preds = self.model_(torch.tensor(X, dtype=torch.float32).to(self.device)).mean(
            dim=1).cpu().numpy()
        return np.clip(preds, 0.0, 6.5)


# ==========================================
# 📥 页面配置与缓存加载模型 (提升网页流畅度)
# ==========================================
st.set_page_config(page_title="多糖基材料 AI 逆向设计平台", page_icon="🧬", layout="wide")


@st.cache_resource
def load_models_and_features():
    try:
        data = joblib.load('model_artifacts_v3.pkl')
        return data['models'], data['X'], data['X'].columns.tolist()
    except FileNotFoundError:
        st.error("❌ 未找到 'model_artifacts_v3.pkl'，请确保文件在同一目录下。")
        st.stop()


models, X_train, X_cols = load_models_and_features()

xgb_model = models['XGBoost']
feat_imps = pd.Series(xgb_model.feature_importances_, index=X_cols).sort_values(ascending=False)
top_20_features = feat_imps.head(20).index.tolist()

background_defaults = {}


def is_binary_col(col_name):
    if any(k in col_name.lower() for k in
           ['concentration', 'dose', 'area', 'porosity', 'weight', 'charge', 'ratio', 'time', 'ph']): return False
    unique_vals = set(X_train[col_name].dropna().unique())
    return unique_vals.issubset({0, 1, 0.0, 1.0})


for col in X_cols:
    if is_binary_col(col):
        background_defaults[col] = float(X_train[col].mode()[0])
    else:
        background_defaults[col] = float(X_train[col].mean())

# ==========================================
# 🎯 智能特征分拣
# ==========================================
ui_dom_features, ui_exp_features, ui_phys_features, ui_fg_features = [], [], [], []
needs_c0_dose = False

for col in X_cols:
    if 'molecular' in col.lower() or 'weight' in col.lower(): continue
    if 'DOM' in col.upper(): ui_dom_features.append(col); continue
    if col == 'Log_C0_to_Dose_Ratio': needs_c0_dose = True; continue
    if col.startswith('FG_'):
        if any(k in col for k in ['FG_Carboxyl', 'FG_Amino', 'FG_Hydroxyl']) or col in top_20_features:
            ui_fg_features.append(col)
        continue
    if any(k in col.lower() for k in ['ph', 'temperature', 'time', 'equilibrium']):
        ui_exp_features.append(col)
    else:
        ui_phys_features.append(col)

# ==========================================
# 🎨 构建 Web 界面
# ==========================================
st.title("🧬 多糖基材料 AI 预测与机理验证平台")
st.markdown(
    "通过输入水体环境、实验参数与材料物理化学表征，基于多模型矩阵（梯度提升树与深度学习消融架构）实时推理最大吸附量 $Q_m$。")
st.divider()

# 使用字典保存用户输入
user_inputs = {}
c0_val, dose_val = 100.0, 0.5

# 布局设计：左侧为输入区，右侧为结果区
col_input, col_result = st.columns([2, 1], gap="large")

with col_input:
    st.subheader("🧪 调节实验与材料参数")

    # 选项卡设计，使界面极其清爽
    tab1, tab2, tab3, tab4 = st.tabs(["💧 水体环境与 DOM", "⚗️ 宏观实验条件", "🧱 物理微观表征", "🧬 表面官能团"])

    with tab1:
        for col in ui_dom_features:
            is_log = col.startswith('Log_')
            display_name = col.replace('Log_', '')
            if is_binary_col(col):
                user_inputs[col] = st.toggle(display_name, value=bool(background_defaults[col]))
            else:
                default_val = np.expm1(background_defaults[col]) if is_log else background_defaults[col]
                user_inputs[col] = st.number_input(display_name, value=float(f"{default_val:.2f}"), step=1.0)

    with tab2:
        if needs_c0_dose:
            col2a, col2b = st.columns(2)
            with col2a: c0_val = st.number_input("Initial Concentration (mg/L)", value=100.0, step=10.0)
            with col2b: dose_val = st.number_input("Adsorbent Dose (mg/ml)", value=0.5, step=0.1)
        for col in ui_exp_features:
            is_log = col.startswith('Log_')
            display_name = col.replace('Log_', '')
            default_val = np.expm1(background_defaults[col]) if is_log else background_defaults[col]
            user_inputs[col] = st.number_input(display_name, value=float(f"{default_val:.2f}"))

    with tab3:
        for col in ui_phys_features:
            is_log = col.startswith('Log_')
            display_name = col.replace('Log_', '')
            default_val = np.expm1(background_defaults[col]) if is_log else background_defaults[col]
            user_inputs[col] = st.number_input(display_name, value=float(f"{default_val:.2f}"))

    with tab4:
        st.markdown("**(Top Ranked Functional Groups & Mandatory Targets)**")
        # 官能团复选框横向排列更美观
        fg_cols = st.columns(3)
        for i, col in enumerate(ui_fg_features):
            with fg_cols[i % 3]:
                display_name = col.replace('Log_', '')
                user_inputs[col] = st.checkbox(display_name, value=bool(background_defaults[col]))

with col_result:
    st.subheader("📊 AI 推理结果")
    st.info("💡 请在左侧调整参数，随后点击下方按钮进行预测。")

    if st.button("🚀 启动全要素联合推理", use_container_width=True, type="primary"):
        with st.spinner("🤖 正在激活多模型矩阵..."):
            # 组装数据
            final_input = background_defaults.copy()
            if needs_c0_dose:
                final_input['Log_C0_to_Dose_Ratio'] = np.log1p(c0_val / (dose_val + 1e-5))

            for col, val in user_inputs.items():
                if is_binary_col(col):
                    final_input[col] = float(val)
                else:
                    is_log = col.startswith('Log_')
                    final_input[col] = np.log1p(float(val)) if is_log else float(val)

            user_df = pd.DataFrame([final_input])[X_cols]

            # 推理
            preds = {}
            for name, model in models.items():
                try:
                    raw_pred_log = model.predict(user_df)[0]
                    preds[name] = np.expm1(raw_pred_log)
                except Exception as e:
                    pass

            best_model_name = 'CatBoost' if 'CatBoost' in preds else list(preds.keys())[0]
            best_val = preds[best_model_name]

            # 展示核心结果
            st.metric(label=f"🔥 核心推荐参考量 ({best_model_name})", value=f"{best_val:.2f} mg/g")

            st.divider()
            st.markdown("##### 🔬 其他模型交叉验证参考")

            # 以优雅的数据框形式展示对比
            res_list = [{"Model": name, "Predicted Qm (mg/g)": round(val, 2)} for name, val in preds.items() if
                        name != best_model_name]
            st.dataframe(pd.DataFrame(res_list), hide_index=True, use_container_width=True)

            st.success("✅ 推理完成！后台隐性特征已自动使用实验均值补齐。")