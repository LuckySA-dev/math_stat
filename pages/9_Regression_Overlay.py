"""
Page 9: Regression Analysis — Linear / Polynomial / Multiple Regression
กราฟซ้อนเปรียบเทียบทุกวิธี (8 วิธี) + วิเคราะห์เชิงลึก Regression
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
from helpers import inject_css, get_data, get_models

inject_css()
df, monthly = get_data()

st.markdown('<div class="page-title">📐 9. Regression Analysis & All-Model Overlay</div>', unsafe_allow_html=True)
st.markdown("---")

test_months = st.slider("🔧 จำนวนเดือนทดสอบ", 3, 12, 6, key="p9_slider")
train, test, results, metrics_df, model_params = get_models(test_months)

# ──────────────────────────────────────
# CONFIG
# ──────────────────────────────────────
ALL_COLORS = {
    "SMA": "#FF6B6B", "Holt": "#FFEAA7", "ExpSmoothing": "#4ECDC4",
    "ARIMA": "#45B7D1", "Prophet": "#96CEB4",
    "LinearReg": "#E8A0BF", "PolyReg": "#B983FF", "MultiReg": "#94D2BD",
}
ALL_NAMES = {
    "SMA": "SMA (3-month)", "Holt": "Holt (Double Exp)",
    "ExpSmoothing": "Holt-Winters", "ARIMA": "ARIMA(2,1,2)",
    "Prophet": "Prophet",
    "LinearReg": "Linear Regression", "PolyReg": "Polynomial Reg (deg=2)",
    "MultiReg": "Multiple Regression",
}
GROUP_TS = ["SMA", "Holt", "ExpSmoothing", "ARIMA", "Prophet"]
GROUP_REG = ["LinearReg", "PolyReg", "MultiReg"]

# ══════════════════════════════════════════════
# SECTION 1: REGRESSION OVERVIEW
# ══════════════════════════════════════════════
st.markdown("""
<div class="section-box">
<h3>📚 Regression Models คืออะไร?</h3>

**Regression** คือวิธีทางสถิติที่หาความสัมพันธ์ระหว่าง **ตัวแปรอิสระ (X)** กับ **ตัวแปรตาม (Y)**  
ในบริบทนี้ เราใช้ **เวลา (t)** และ **Features อื่นๆ** เพื่อทำนายราคาทองคำ

| วิธี | สมการ | จำนวน Features | จุดเด่น |
|------|-------|---------------|---------|
| **Linear Regression** | $\\hat{y} = β_0 + β_1 t$ | 1 | เรียบง่าย จับ Linear Trend |
| **Polynomial Regression** | $\\hat{y} = β_0 + β_1 t + β_2 t^2$ | 2 | จับ Non-linear Trend (โค้ง) |
| **Multiple Regression** | $\\hat{y} = β_0 + β_1 t + β_2 t^2 + β_3 \\sin + β_4 \\cos$ | 4 | จับทั้ง Trend + Seasonal |
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 2: REGRESSION FIT LINES (Full Training Data)
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 Regression Fit Lines — Full Training Period")
st.markdown("""
<div class="highlight-box">
กราฟด้านล่างแสดง <b>เส้น Fit ของ Regression 3 วิธี</b> บนข้อมูล Training ทั้งหมด<br>
เห็นได้ชัดว่า Linear เป็นเส้นตรง, Polynomial เป็นเส้นโค้ง, Multiple มีทั้งโค้งและ Seasonal
</div>
""", unsafe_allow_html=True)

# Rebuild regression predictions on train for visualization
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from analysis import _build_multi_features

n_train = len(train)
t_train = np.arange(n_train).reshape(-1, 1)
y_train = train["avg_price"].values

# Linear
lr = LinearRegression().fit(t_train, y_train)
lr_train_pred = lr.predict(t_train)

# Polynomial
poly_tf = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly_tf.fit_transform(t_train)
pr = LinearRegression().fit(X_poly_train, y_train)
pr_train_pred = pr.predict(X_poly_train)

# Multiple
X_multi_train = _build_multi_features(train.index, start_idx=0)
mr = LinearRegression().fit(X_multi_train, y_train)
mr_train_pred = mr.predict(X_multi_train)

fig_fit = go.Figure()
fig_fit.add_trace(go.Scatter(
    x=train.index, y=y_train,
    mode="lines", name="Actual (Train)",
    line=dict(color="#FFD700", width=2),
))
for name, pred, color, dash in [
    ("Linear Regression", lr_train_pred, "#E8A0BF", "dash"),
    ("Polynomial Reg (deg=2)", pr_train_pred, "#B983FF", "dot"),
    ("Multiple Regression", mr_train_pred, "#94D2BD", "dashdot"),
]:
    fig_fit.add_trace(go.Scatter(
        x=train.index, y=pred,
        mode="lines", name=name,
        line=dict(color=color, width=2.5, dash=dash),
    ))

fig_fit.update_layout(
    template="plotly_dark", height=500,
    title="Regression Fit Lines on Training Data",
    xaxis_title="Date", yaxis_title="Price (USD)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig_fit, use_container_width=True)

# R² cards
st.markdown("### 📊 R² (Coefficient of Determination) — Training Fit Quality")
c1, c2, c3 = st.columns(3)
r2_data = [
    ("Linear Regression", model_params["LinearReg"]["r2_train"], "#E8A0BF"),
    ("Polynomial Reg", model_params["PolyReg"]["r2_train"], "#B983FF"),
    ("Multiple Regression", model_params["MultiReg"]["r2_train"], "#94D2BD"),
]
for col, (name, r2, color) in zip([c1, c2, c3], r2_data):
    with col:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 2px solid {color};
                    border-radius: 12px; padding: 1rem; text-align: center;">
            <div style="font-size:0.9rem; color:#AAA;">{name}</div>
            <div style="font-size:2.2rem; font-weight:800; color:{color};">{r2:.4f}</div>
            <div style="font-size:0.85rem; color:#888;">R² = อธิบายความแปรปรวนได้ {r2*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 3: ALL 8 METHODS OVERLAY (Test Period)
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔀 กราฟซ้อนทุกวิธี — All 8 Methods Overlay")
st.markdown("""
<div class="highlight-box">
กราฟด้านล่างซ้อน <b>ราคาจริง</b> กับ <b>ราคาทำนายจากทั้ง 8 วิธี</b> ในช่วง Test Period<br>
ใช้สำหรับเปรียบเทียบว่าแต่ละวิธีทำนายออกมาใกล้เคียงราคาจริงแค่ไหน<br>
<b>สีทอง (ทึบ)</b> = ราคาจริง &emsp; <b>สีอื่น (ประ)</b> = แต่ละวิธี
</div>
""", unsafe_allow_html=True)

test_dates = test.index.strftime("%Y-%m-%d").tolist()
actual_vals = [round(v, 2) for v in test["avg_price"].values]

all_series = [{
    "name": "Actual (ราคาจริง)",
    "type": "line",
    "data": actual_vals,
    "lineStyle": {"color": "#FFD700", "width": 3.5},
    "symbol": "circle", "symbolSize": 10,
    "itemStyle": {"color": "#FFD700"},
    "z": 10,
}]

for key in list(results.keys()):
    pred = results[key]
    all_series.append({
        "name": ALL_NAMES.get(key, key),
        "type": "line",
        "data": [round(v, 2) for v in pred.values],
        "lineStyle": {"color": ALL_COLORS.get(key, "#888"), "width": 2, "type": "dashed"},
        "symbol": "diamond", "symbolSize": 6,
        "itemStyle": {"color": ALL_COLORS.get(key, "#888")},
    })

overlay_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {
        "data": ["Actual (ราคาจริง)"] + [ALL_NAMES.get(k, k) for k in results.keys()],
        "textStyle": {"color": "#CCC", "fontSize": 11},
        "bottom": 0, "type": "scroll",
    },
    "grid": {"top": 40, "bottom": 80, "left": 70, "right": 30},
    "xAxis": {"type": "category", "data": test_dates, "boundaryGap": False},
    "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
    "series": all_series,
    "backgroundColor": "transparent",
}
st_echarts(options=overlay_opt, height="550px")

# ══════════════════════════════════════════════
# SECTION 4: SPLIT VIEW — Time Series vs Regression
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚔️ Time Series Methods vs Regression Methods")
st.markdown("""
<div class="section-box">
แบ่งกราฟเป็น 2 กลุ่มเพื่อเปรียบเทียบอย่างชัดเจน:<br>
• <b>กลุ่ม Time Series</b> (5 วิธี): SMA, Holt, Holt-Winters, ARIMA, Prophet<br>
• <b>กลุ่ม Regression</b> (3 วิธี): Linear, Polynomial, Multiple
</div>
""", unsafe_allow_html=True)

col_ts, col_reg = st.columns(2)

with col_ts:
    st.markdown("### 📊 Time Series (5 วิธี)")
    ts_series = [{
        "name": "Actual",
        "type": "line",
        "data": actual_vals,
        "lineStyle": {"color": "#FFD700", "width": 3},
        "symbol": "circle", "symbolSize": 8,
        "itemStyle": {"color": "#FFD700"},
    }]
    for key in GROUP_TS:
        if key in results:
            ts_series.append({
                "name": ALL_NAMES[key],
                "type": "line",
                "data": [round(v, 2) for v in results[key].values],
                "lineStyle": {"color": ALL_COLORS[key], "width": 2, "type": "dashed"},
                "symbol": "diamond", "symbolSize": 5,
                "itemStyle": {"color": ALL_COLORS[key]},
            })
    ts_opt = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Actual"] + [ALL_NAMES[k] for k in GROUP_TS if k in results],
                   "textStyle": {"color": "#CCC", "fontSize": 10}, "bottom": 0},
        "xAxis": {"type": "category", "data": test_dates, "boundaryGap": False},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
        "series": ts_series,
        "backgroundColor": "transparent",
    }
    st_echarts(options=ts_opt, height="420px")

with col_reg:
    st.markdown("### 📐 Regression (3 วิธี)")
    reg_series = [{
        "name": "Actual",
        "type": "line",
        "data": actual_vals,
        "lineStyle": {"color": "#FFD700", "width": 3},
        "symbol": "circle", "symbolSize": 8,
        "itemStyle": {"color": "#FFD700"},
    }]
    for key in GROUP_REG:
        if key in results:
            reg_series.append({
                "name": ALL_NAMES[key],
                "type": "line",
                "data": [round(v, 2) for v in results[key].values],
                "lineStyle": {"color": ALL_COLORS[key], "width": 2, "type": "dashed"},
                "symbol": "diamond", "symbolSize": 5,
                "itemStyle": {"color": ALL_COLORS[key]},
            })
    reg_opt = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["Actual"] + [ALL_NAMES[k] for k in GROUP_REG if k in results],
                   "textStyle": {"color": "#CCC", "fontSize": 10}, "bottom": 0},
        "xAxis": {"type": "category", "data": test_dates, "boundaryGap": False},
        "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
        "series": reg_series,
        "backgroundColor": "transparent",
    }
    st_echarts(options=reg_opt, height="420px")

# ══════════════════════════════════════════════
# SECTION 5: FULL METRICS TABLE (all 8)
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📋 ตาราง Metrics ทั้ง 8 วิธี (เรียงจาก MAPE ต่ำสุด)")

st.dataframe(
    metrics_df.style
    .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="rgba(45,106,79,0.4)")
    .highlight_max(subset=["MAE", "RMSE", "MAPE (%)"], color="rgba(255,107,107,0.3)")
    .format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE (%)": "{:.2f}%"}),
    use_container_width=True, hide_index=True,
)

# ══════════════════════════════════════════════
# SECTION 6: MAPE BAR — all 8
# ══════════════════════════════════════════════
st.markdown("### 📊 MAPE Comparison — ทุกวิธี")

bar_colors_map = {
    "Holt": "#FFEAA7", "Holt-Winters": "#4ECDC4", "ARIMA": "#45B7D1",
    "SMA": "#FF6B6B", "Prophet": "#96CEB4",
    "Linear Regression": "#E8A0BF", "Polynomial": "#B983FF",
    "Multiple Regression": "#94D2BD",
}
mape_bar_data = []
for _, row in metrics_df.iterrows():
    c = "#888"
    for k, v in bar_colors_map.items():
        if k in row["method"]:
            c = v
            break
    mape_bar_data.append({"value": row["MAPE (%)"], "itemStyle": {"color": c}})

mape_all_opt = {
    "tooltip": {"trigger": "axis", "formatter": "{b}: {c}%"},
    "xAxis": {"type": "category", "data": metrics_df["method"].tolist(),
              "axisLabel": {"rotate": 25, "fontSize": 10}},
    "yAxis": {"type": "value", "name": "MAPE (%)", "axisLabel": {"formatter": "{value}%"}},
    "series": [{
        "type": "bar",
        "data": mape_bar_data,
        "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 12, "fontWeight": "bold"},
        "barWidth": "50%",
        "itemStyle": {"borderRadius": [8, 8, 0, 0]},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=mape_all_opt, height="450px")

# ══════════════════════════════════════════════
# SECTION 7: REGRESSION EQUATIONS SUMMARY
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📐 Regression — สมการที่ Fit ได้")

mp_lr = model_params["LinearReg"]
mp_pr = model_params["PolyReg"]
mp_mr = model_params["MultiReg"]

st.markdown(f"""
<div style="background: rgba(232,160,191,0.08); border: 2px solid rgba(232,160,191,0.3);
            border-radius: 14px; padding: 1.3rem; margin: 0.8rem 0;">
<h4 style="color:#E8A0BF; margin-top:0;">📏 Linear Regression</h4>

$$\\hat{{y}}_t = {mp_lr['intercept']:,.2f} + {mp_lr['coef_t']:,.4f} \\cdot t$$

- **β₁ = {mp_lr['coef_t']:,.4f}** → ราคาทองเพิ่มขึ้นเฉลี่ย **~${mp_lr['coef_t']:,.2f}/เดือน**
- **R² = {mp_lr['r2_train']:.4f}** → อธิบายความแปรปรวนได้ {mp_lr['r2_train']*100:.1f}%
</div>
""", unsafe_allow_html=True)

coefs_poly = mp_pr["coefs"]
st.markdown(f"""
<div style="background: rgba(185,131,255,0.08); border: 2px solid rgba(185,131,255,0.3);
            border-radius: 14px; padding: 1.3rem; margin: 0.8rem 0;">
<h4 style="color:#B983FF; margin-top:0;">📈 Polynomial Regression (Degree 2)</h4>

$$\\hat{{y}}_t = {mp_pr['intercept']:,.2f} + {coefs_poly[0]:,.4f} \\cdot t + {coefs_poly[1]:,.6f} \\cdot t^2$$

- **β₂ = {coefs_poly[1]:,.6f}** → {"โค้งเปิดขึ้น (ราคาเร่งตัว)" if coefs_poly[1] > 0 else "โค้งเปิดลง (ราคาชะลอตัว)"}
- **R² = {mp_pr['r2_train']:.4f}** → อธิบายความแปรปรวนได้ {mp_pr['r2_train']*100:.1f}%
</div>
""", unsafe_allow_html=True)

coefs_mr = mp_mr["coefs"]
st.markdown(f"""
<div style="background: rgba(148,210,189,0.08); border: 2px solid rgba(148,210,189,0.3);
            border-radius: 14px; padding: 1.3rem; margin: 0.8rem 0;">
<h4 style="color:#94D2BD; margin-top:0;">🔀 Multiple Regression</h4>

$$\\hat{{y}}_t = {mp_mr['intercept']:,.2f} + {coefs_mr['t']:,.4f} \\cdot t + {coefs_mr['t²']:,.6f} \\cdot t^2 + {coefs_mr['sin(month)']:,.2f}\\sin\\!\\left(\\frac{{2\\pi m}}{{12}}\\right) + {coefs_mr['cos(month)']:,.2f}\\cos\\!\\left(\\frac{{2\\pi m}}{{12}}\\right)$$

- **Features:** Time + Time² + Seasonal (sin/cos encoding)
- **R² = {mp_mr['r2_train']:.4f}** → อธิบายความแปรปรวนได้ {mp_mr['r2_train']*100:.1f}%
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SECTION 8: REGRESSION vs TIME SERIES — SUMMARY
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 💡 สรุปเปรียบเทียบ: Time Series vs Regression")

# Compute group averages
ts_mape = []
reg_mape = []
match_ts = {"SMA", "Holt", "Holt-Winters", "ARIMA", "Prophet"}
match_reg = {"Linear Regression", "Polynomial", "Multiple Regression"}
for _, row in metrics_df.iterrows():
    name = row["method"]
    for k in match_ts:
        if k in name:
            ts_mape.append(row["MAPE (%)"])
            break
    for k in match_reg:
        if k in name:
            reg_mape.append(row["MAPE (%)"])
            break

avg_ts = np.mean(ts_mape) if ts_mape else 0
avg_reg = np.mean(reg_mape) if reg_mape else 0
best_ts = min(ts_mape) if ts_mape else 0
best_reg = min(reg_mape) if reg_mape else 0

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div style="background: rgba(255,215,0,0.06); border: 2px solid rgba(255,215,0,0.25);
                border-radius: 14px; padding: 1.5rem; text-align: center;">
        <h3 style="color:#FFD700; margin-top:0;">📊 Time Series (5 วิธี)</h3>
        <div style="font-size:1.1rem; color:#CCC;">MAPE เฉลี่ย: <b>{avg_ts:.2f}%</b></div>
        <div style="font-size:1.1rem; color:#81C784;">Best MAPE: <b>{best_ts:.2f}%</b></div>
        <hr style="border-color: rgba(255,215,0,0.15);">
        <div style="color:#AAA; font-size:0.9rem;">
            ✅ จับ Autocorrelation ได้<br>
            ✅ ออกแบบมาสำหรับข้อมูลอนุกรมเวลา<br>
            ⚠️ ต้อง Tune Parameters
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: rgba(185,131,255,0.06); border: 2px solid rgba(185,131,255,0.25);
                border-radius: 14px; padding: 1.5rem; text-align: center;">
        <h3 style="color:#B983FF; margin-top:0;">📐 Regression (3 วิธี)</h3>
        <div style="font-size:1.1rem; color:#CCC;">MAPE เฉลี่ย: <b>{avg_reg:.2f}%</b></div>
        <div style="font-size:1.1rem; color:#81C784;">Best MAPE: <b>{best_reg:.2f}%</b></div>
        <hr style="border-color: rgba(185,131,255,0.15);">
        <div style="color:#AAA; font-size:0.9rem;">
            ✅ เข้าใจง่าย มี R² วัดคุณภาพ<br>
            ✅ ใส่ Features เพิ่มได้ง่าย<br>
            ⚠️ ไม่จับ Autocorrelation โดยตรง
        </div>
    </div>
    """, unsafe_allow_html=True)

winner_group = "Time Series" if avg_ts < avg_reg else "Regression"
st.markdown(f"""
<div class="highlight-box" style="text-align: center; font-size: 1.1rem;">
<b>🏆 กลุ่มที่ได้ MAPE เฉลี่ยต่ำกว่า: <span style="color:#FFD700;">{winner_group}</span></b><br><br>
<b>สรุป:</b> วิธี Time Series (โดยเฉพาะ Holt) ทำงานได้ดีกว่า Regression สำหรับข้อมูลทองคำ<br>
เพราะราคาทองคำมี <b>Strong Autocorrelation</b> — ราคาวันนี้ขึ้นอยู่กับราคาในอดีต<br>
ซึ่ง Time Series Methods ถูกออกแบบมาจับ Pattern เหล่านี้โดยเฉพาะ<br><br>
อย่างไรก็ตาม <b>Multiple Regression</b> ที่มี Seasonal Features ทำได้ใกล้เคียง<br>
เพราะจับทั้ง Trend (ผ่าน t, t²) และ Seasonal (ผ่าน sin/cos) ได้
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("หน้า 9/9 — Regression Analysis & All-Model Overlay")
