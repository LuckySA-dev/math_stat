"""
Page 8: เปรียบเทียบ 3 วิธีเด่น — Slide-Ready
เปรียบเทียบเฉพาะ Top 3 วิธี (Holt, Holt-Winters, ARIMA) ที่ให้ MAPE ต่ำสุด
ออกแบบสำหรับนำเสนอ (Presentation / Slide)
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

# ──────────────────────────────────────
# Extra CSS for slide-ready look
# ──────────────────────────────────────
st.markdown("""
<style>
    .slide-title {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.5rem; font-weight: 800; text-align: center;
        padding: 1rem 0 0.3rem 0; letter-spacing: 0.5px;
    }
    .slide-subtitle {
        color: #9CA3AF; text-align: center; font-size: 1.15rem;
        margin-bottom: 1.5rem;
    }
    .method-card {
        border-radius: 14px; padding: 1.3rem; margin: 0.5rem 0;
        text-align: center;
    }
    .method-card h2 { margin: 0 0 0.3rem 0; font-size: 1.6rem; }
    .method-card .mape-val { font-size: 2.2rem; font-weight: 800; }
    .slide-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,215,0,0.12);
        border-radius: 14px; padding: 1.5rem; margin: 1.2rem 0;
    }
    .vs-badge {
        display: inline-block; background: rgba(255,215,0,0.15);
        border: 1px solid rgba(255,215,0,0.3); border-radius: 20px;
        padding: 0.2rem 1rem; font-weight: 700; color: #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# DATA
# ──────────────────────────────────────
test_months = st.slider("🔧 จำนวนเดือนทดสอบ", 3, 12, 6, key="p8_slider")
train, test, results, metrics_df, model_params = get_models(test_months)

# Top 3 method keys (sorted by MAPE)
TOP3_KEYS = ["Holt", "ExpSmoothing", "ARIMA"]
TOP3_NAMES = {
    "Holt": "Holt (Double Exp. Smoothing)",
    "ExpSmoothing": "Holt-Winters (Triple Exp. Smoothing)",
    "ARIMA": "ARIMA(2,1,2)",
}
TOP3_COLORS = {
    "Holt": "#FFEAA7",
    "ExpSmoothing": "#4ECDC4",
    "ARIMA": "#45B7D1",
}
TOP3_ICONS = {"Holt": "🥇", "ExpSmoothing": "🥈", "ARIMA": "🥉"}


# Get metrics for top 3 only
match_map = {"Holt": "Holt", "ExpSmoothing": "Holt-Winters", "ARIMA": "ARIMA"}
top3_metrics = []
for k in TOP3_KEYS:
    row = metrics_df[metrics_df["method"].str.contains(match_map[k])]
    if not row.empty:
        top3_metrics.append({"key": k, **row.iloc[0].to_dict()})
top3_df = pd.DataFrame(top3_metrics)

# ══════════════════════════════════════════════
# SLIDE 1: TITLE
# ══════════════════════════════════════════════
st.markdown('<div class="slide-title">⚔️ เปรียบเทียบ 3 วิธีเด่น</div>', unsafe_allow_html=True)
st.markdown('<div class="slide-subtitle">Holt vs Holt-Winters vs ARIMA — Top 3 วิธีที่ให้ MAPE ต่ำที่สุด</div>', unsafe_allow_html=True)

st.markdown("""
<div class="slide-section">
<h3 style="text-align:center; margin-top:0;">🎯 ทำไมเลือก 3 วิธีนี้?</h3>
<p style="text-align:center; color:#CCC; font-size:1.05rem;">
จาก 5 วิธีที่ทดสอบ (SMA, Holt, Holt-Winters, ARIMA, Prophet)<br>
3 วิธีที่ให้ค่า <b style="color:#FFD700;">MAPE ต่ำที่สุด</b> ถูกเลือกมาเปรียบเทียบอย่างละเอียด<br>
เพราะทั้ง 3 วิธีมีค่า MAPE <b>ใกล้เคียงกันมาก</b> (ห่างกันไม่ถึง 1%)
</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SLIDE 2: METHOD CARDS — MAPE Overview
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 ภาพรวม MAPE ของ 3 วิธี")

cols = st.columns(3)
card_bgs = [
    "linear-gradient(135deg, rgba(255,234,167,0.12), rgba(255,234,167,0.03))",
    "linear-gradient(135deg, rgba(78,205,196,0.12), rgba(78,205,196,0.03))",
    "linear-gradient(135deg, rgba(69,183,209,0.12), rgba(69,183,209,0.03))",
]
card_borders = ["rgba(255,234,167,0.3)", "rgba(78,205,196,0.3)", "rgba(69,183,209,0.3)"]

for i, k in enumerate(TOP3_KEYS):
    m = top3_df[top3_df["key"] == k]
    if m.empty:
        continue
    r = m.iloc[0]
    with cols[i]:
        st.markdown(f"""
        <div class="method-card" style="background: {card_bgs[i]}; border: 2px solid {card_borders[i]};">
            <h2>{TOP3_ICONS[k]} {TOP3_NAMES[k]}</h2>
            <div class="mape-val" style="color: {TOP3_COLORS[k]};">{r['MAPE (%)']:.2f}%</div>
            <p style="color:#AAA; margin:0.3rem 0 0 0;">MAPE</p>
            <hr style="border-color: {card_borders[i]}; margin: 0.8rem 0;">
            <span style="color:#4ECDC4;">MAE: ${r['MAE']:,.2f}</span> &nbsp;
            <span style="color:#FF6B6B;">RMSE: ${r['RMSE']:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SLIDE 3: COMBINED LINE CHART — Actual vs Top 3
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 กราฟเปรียบเทียบ: ราคาจริง vs 3 วิธีทำนาย")
st.markdown("""
<div class="highlight-box">
กราฟด้านล่างแสดงเฉพาะ <b>ช่วง Test Period</b> เพื่อเปรียบเทียบให้เห็นชัด<br>
<b>เส้นทึบสีทอง</b> = ราคาจริง &emsp; <b>เส้นประ</b> = ราคาทำนายจากแต่ละวิธี
</div>
""", unsafe_allow_html=True)

test_dates = test.index.strftime("%Y-%m-%d").tolist()
actual_vals = [round(v, 2) for v in test["avg_price"].values]

echarts_top3_series = [
    {
        "name": "Actual (ราคาจริง)",
        "type": "line",
        "data": actual_vals,
        "lineStyle": {"color": "#FFD700", "width": 3},
        "symbol": "circle", "symbolSize": 10,
        "itemStyle": {"color": "#FFD700"},
    }
]
for k in TOP3_KEYS:
    pred = results[k]
    echarts_top3_series.append({
        "name": TOP3_NAMES[k],
        "type": "line",
        "data": [round(v, 2) for v in pred.values],
        "lineStyle": {"color": TOP3_COLORS[k], "width": 2.5, "type": "dashed"},
        "symbol": "diamond", "symbolSize": 9,
        "itemStyle": {"color": TOP3_COLORS[k]},
    })

top3_line_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {
        "data": ["Actual (ราคาจริง)"] + [TOP3_NAMES[k] for k in TOP3_KEYS],
        "textStyle": {"color": "#CCC", "fontSize": 12},
        "bottom": 0,
    },
    "grid": {"top": 40, "bottom": 60, "left": 70, "right": 30},
    "xAxis": {"type": "category", "data": test_dates, "boundaryGap": False,
              "axisLabel": {"fontSize": 12}},
    "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}", "fontSize": 12}},
    "series": echarts_top3_series,
    "backgroundColor": "transparent",
}
st_echarts(options=top3_line_opt, height="480px")

# ══════════════════════════════════════════════
# SLIDE 4: MONTH-BY-MONTH TABLE
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📋 ตารางเปรียบเทียบรายเดือน")
st.markdown("""
<div class="highlight-box">
ตารางแสดงราคาจริง vs ทำนายรายเดือนของทั้ง 3 วิธี พร้อม Error (%) ของแต่ละวิธี
</div>
""", unsafe_allow_html=True)

table_data = {"เดือน": test.index.strftime("%b %Y"), "ราคาจริง ($)": test["avg_price"].round(2).values}
for k in TOP3_KEYS:
    pred = results[k]
    table_data[f"{TOP3_NAMES[k]} ($)"] = pred.round(2).values
    err_pct = ((test["avg_price"].values - pred.values) / test["avg_price"].values * 100)
    table_data[f"Error {k} (%)"] = err_pct.round(2)

table_df = pd.DataFrame(table_data)
st.dataframe(table_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# SLIDE 5: BAR CHARTS — MAPE / MAE / RMSE
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 เปรียบเทียบ Metrics: MAPE · MAE · RMSE")

col_mape, col_mae_rmse = st.columns([1, 1])

with col_mape:
    st.markdown("### MAPE (%) — ยิ่งน้อยยิ่งดี")
    mape_vals = []
    for k in TOP3_KEYS:
        m = top3_df[top3_df["key"] == k]
        mape_vals.append({
            "value": round(m.iloc[0]["MAPE (%)"], 2),
            "itemStyle": {"color": TOP3_COLORS[k]},
        })

    mape_bar = {
        "tooltip": {"trigger": "axis", "formatter": "{b}: {c}%"},
        "xAxis": {"type": "category",
                  "data": [TOP3_NAMES[k] for k in TOP3_KEYS],
                  "axisLabel": {"fontSize": 11, "rotate": 15}},
        "yAxis": {"type": "value", "name": "MAPE (%)",
                  "axisLabel": {"formatter": "{value}%"}},
        "series": [{
            "type": "bar", "data": mape_vals,
            "label": {"show": True, "position": "top", "formatter": "{c}%",
                      "fontSize": 14, "fontWeight": "bold"},
            "barWidth": "55%",
            "itemStyle": {"borderRadius": [8, 8, 0, 0]},
        }],
        "backgroundColor": "transparent",
    }
    st_echarts(options=mape_bar, height="380px")

with col_mae_rmse:
    st.markdown("### MAE & RMSE (USD) — ยิ่งน้อยยิ่งดี")
    mae_vals = []
    rmse_vals = []
    for k in TOP3_KEYS:
        m = top3_df[top3_df["key"] == k]
        mae_vals.append(round(m.iloc[0]["MAE"], 2))
        rmse_vals.append(round(m.iloc[0]["RMSE"], 2))

    mae_rmse_bar = {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["MAE", "RMSE"], "textStyle": {"color": "#CCC"}},
        "xAxis": {"type": "category",
                  "data": [TOP3_NAMES[k] for k in TOP3_KEYS],
                  "axisLabel": {"fontSize": 11, "rotate": 15}},
        "yAxis": {"type": "value", "name": "USD",
                  "axisLabel": {"formatter": "${value}"}},
        "series": [
            {"name": "MAE", "type": "bar", "data": mae_vals,
             "itemStyle": {"color": "#4ECDC4", "borderRadius": [6, 6, 0, 0]},
             "label": {"show": True, "position": "top", "fontSize": 11}},
            {"name": "RMSE", "type": "bar", "data": rmse_vals,
             "itemStyle": {"color": "#FF6B6B", "borderRadius": [6, 6, 0, 0]},
             "label": {"show": True, "position": "top", "fontSize": 11}},
        ],
        "backgroundColor": "transparent",
    }
    st_echarts(options=mae_rmse_bar, height="380px")

# ══════════════════════════════════════════════
# SLIDE 6: ERROR DISTRIBUTION (Plotly box plot)
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📦 การกระจายตัวของ Error — Box Plot")
st.markdown("""
<div class="highlight-box">
Box Plot แสดง <b>การกระจายตัวของค่า Error (%)</b> ของแต่ละวิธี<br>
กล่องแคบ = Error สม่ำเสมอ (เสถียร) &emsp; กล่องกว้าง = Error ผันผวน
</div>
""", unsafe_allow_html=True)

fig_box = go.Figure()
for k in TOP3_KEYS:
    pred = results[k]
    err_pct = (test["avg_price"].values - pred.values) / test["avg_price"].values * 100
    fig_box.add_trace(go.Box(
        y=err_pct, name=TOP3_NAMES[k],
        marker_color=TOP3_COLORS[k],
        boxmean="sd",
    ))
fig_box.update_layout(
    template="plotly_dark", height=420,
    yaxis_title="Error (%)",
    showlegend=False,
    title="Distribution of Forecast Error (%) — Top 3 Methods",
)
st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════════════════════════════════
# SLIDE 7: INDIVIDUAL OVERLAY (Plotly with confidence band)
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔎 กราฟแต่ละวิธี + Confidence Band")
st.markdown("""
<div class="highlight-box">
แสดง Full Timeline (Train + Test) พร้อม <b>แถบความเชื่อมั่น ±1σ</b> ของแต่ละวิธี
</div>
""", unsafe_allow_html=True)

fig_panels = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=[TOP3_NAMES[k] for k in TOP3_KEYS],
    vertical_spacing=0.08,
)

for i, k in enumerate(TOP3_KEYS, 1):
    pred = results[k]
    error = test["avg_price"].values - pred.values
    err_std = np.std(error)

    # Actual
    fig_panels.add_trace(go.Scatter(
        x=monthly.index, y=monthly["avg_price"],
        mode="lines", name="Actual" if i == 1 else None,
        line=dict(color="#FFD700", width=1.5),
        showlegend=(i == 1),
    ), row=i, col=1)

    # Prediction
    fig_panels.add_trace(go.Scatter(
        x=test.index, y=pred.values,
        mode="lines+markers",
        name=TOP3_NAMES[k] if i == 1 else None,
        line=dict(color=TOP3_COLORS[k], width=2.5, dash="dash"),
        marker=dict(size=7, symbol="diamond"),
        showlegend=(i == 1),
    ), row=i, col=1)

    # Confidence band ±1σ
    fig_panels.add_trace(go.Scatter(
        x=list(test.index) + list(test.index[::-1]),
        y=list(pred.values + err_std) + list((pred.values - err_std)[::-1]),
        fill="toself", fillcolor=f"rgba{tuple(list(bytes.fromhex(TOP3_COLORS[k].lstrip('#'))) + [0.12])}",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
    ), row=i, col=1)

fig_panels.update_layout(
    template="plotly_dark", height=900,
    title_text="Top 3 Methods: Full Timeline + Confidence Band (±1σ)",
)
st.plotly_chart(fig_panels, use_container_width=True)

# ══════════════════════════════════════════════
# SLIDE 8: EXPLANATION & SUMMARY
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📝 อธิบายแต่ละวิธี")

# Holt
st.markdown(f"""
<div class="slide-section" style="border-left: 4px solid {TOP3_COLORS['Holt']};">
<h3 style="color:{TOP3_COLORS['Holt']}; margin-top:0;">🥇 Holt (Double Exponential Smoothing)</h3>

**หลักการ:** ใช้การปรับเรียบแบบ 2 ชั้น — จับ **Level** (ระดับ) และ **Trend** (แนวโน้ม)

$$\\hat{{y}}_{{t+h}} = l_t + h \\cdot b_t$$

- $l_t$ = Level — ระดับปัจจุบันของข้อมูล
- $b_t$ = Trend — อัตราการเปลี่ยนแปลงต่อเดือน
- $h$ = จำนวนเดือนที่ทำนายล่วงหน้า

**ทำไมได้ MAPE ต่ำสุด:**
- ราคาทองคำมี **Strong Uptrend** → Holt จับ Trend ได้ดีที่สุด
- ไม่มี Seasonal Component ที่ซับซ้อนเกินไป ทำให้ไม่ Overfit
- พารามิเตอร์น้อย (α, β เท่านั้น) → เสถียรกว่าวิธีที่ซับซ้อน
</div>
""", unsafe_allow_html=True)

# Holt-Winters
st.markdown(f"""
<div class="slide-section" style="border-left: 4px solid {TOP3_COLORS['ExpSmoothing']};">
<h3 style="color:{TOP3_COLORS['ExpSmoothing']}; margin-top:0;">🥈 Holt-Winters (Triple Exponential Smoothing)</h3>

**หลักการ:** เพิ่ม **Seasonal Component** เข้ามาจาก Holt

$$\\hat{{y}}_{{t+h}} = l_t + h \\cdot b_t + s_{{t+h-m}}$$

- $s_t$ = Seasonal Component — รูปแบบที่ซ้ำทุก $m$ เดือน
- ใช้ Additive Model, $m = 12$ (วัฏจักรรายปี)

**เหตุผลที่ได้อันดับ 2:**
- Seasonal Pattern ของทองคำมีไม่มากนัก
- การเพิ่มพารามิเตอร์ γ ทำให้ซับซ้อนขึ้นเล็กน้อย
- แต่ยังจับ Trend ได้ดีเกือบเท่า Holt
</div>
""", unsafe_allow_html=True)

# ARIMA
st.markdown(f"""
<div class="slide-section" style="border-left: 4px solid {TOP3_COLORS['ARIMA']};">
<h3 style="color:{TOP3_COLORS['ARIMA']}; margin-top:0;">🥉 ARIMA(2,1,2)</h3>

**หลักการ:** รวม 3 ส่วน — AutoRegressive (AR), Integrated (I), Moving Average (MA)

$$(1 - \\phi_1 B - \\phi_2 B^2)(1-B)y_t = (1 + \\theta_1 B + \\theta_2 B^2)\\varepsilon_t$$

- $p=2$ → ใช้ข้อมูล 2 เดือนย้อนหลัง (AR)
- $d=1$ → Differencing 1 ครั้งเพื่อให้ Stationary
- $q=2$ → ใช้ 2 Error Terms ย้อนหลัง (MA)

**เหตุผลที่ได้อันดับ 3:**
- ARIMA ยืดหยุ่นสูง แต่ต้อง Tune Parameters (p,d,q)
- ข้อมูลทองคำมี Structural Break (COVID, 2024 surge) ที่ ARIMA จับได้ไม่ดีนัก
- MAPE ห่างจาก Holt-Winters เพียงเล็กน้อย
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# SLIDE 9: FINAL VERDICT
# ══════════════════════════════════════════════
st.markdown("---")

# Get actual MAPE values
holt_mape = top3_df[top3_df["key"] == "Holt"].iloc[0]["MAPE (%)"]
hw_mape = top3_df[top3_df["key"] == "ExpSmoothing"].iloc[0]["MAPE (%)"]
arima_mape = top3_df[top3_df["key"] == "ARIMA"].iloc[0]["MAPE (%)"]
diff_hw = hw_mape - holt_mape
diff_arima = arima_mape - holt_mape

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(255,215,0,0.12), rgba(255,165,0,0.05));
            border: 2px solid rgba(255,215,0,0.35); border-radius: 18px;
            padding: 2rem; text-align: center; margin: 1rem 0;">
    <h2 style="margin:0; color: #FFD700;">🏆 สรุปผลการเปรียบเทียบ 3 วิธี</h2>
    <br>
    <table style="margin: auto; border-collapse: collapse; font-size: 1.1rem; width: 90%;">
        <tr style="border-bottom: 2px solid rgba(255,215,0,0.2);">
            <th style="padding: 0.6rem 1.5rem; text-align:left;">อันดับ</th>
            <th style="padding: 0.6rem 1.5rem; text-align:left;">วิธี</th>
            <th style="padding: 0.6rem 1.5rem; text-align:right;">MAPE</th>
            <th style="padding: 0.6rem 1.5rem; text-align:right;">ห่างจากที่ 1</th>
        </tr>
        <tr style="background: rgba(255,234,167,0.08);">
            <td style="padding: 0.6rem 1.5rem;">🥇</td>
            <td style="padding: 0.6rem 1.5rem; color:{TOP3_COLORS['Holt']}; font-weight:700;">Holt</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; font-weight:700;">{holt_mape:.2f}%</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; color:#81C784;">—</td>
        </tr>
        <tr style="background: rgba(78,205,196,0.06);">
            <td style="padding: 0.6rem 1.5rem;">🥈</td>
            <td style="padding: 0.6rem 1.5rem; color:{TOP3_COLORS['ExpSmoothing']}; font-weight:700;">Holt-Winters</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; font-weight:700;">{hw_mape:.2f}%</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; color:#FFB74D;">+{diff_hw:.2f}%</td>
        </tr>
        <tr style="background: rgba(69,183,209,0.06);">
            <td style="padding: 0.6rem 1.5rem;">🥉</td>
            <td style="padding: 0.6rem 1.5rem; color:{TOP3_COLORS['ARIMA']}; font-weight:700;">ARIMA(2,1,2)</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; font-weight:700;">{arima_mape:.2f}%</td>
            <td style="padding: 0.6rem 1.5rem; text-align:right; color:#FFB74D;">+{diff_arima:.2f}%</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="slide-section">
<h3 style="margin-top:0;">💡 บทสรุปสำหรับการนำเสนอ</h3>

**1. ผู้ชนะคือ Holt (Double Exponential Smoothing)** ด้วย MAPE เพียง **{holt_mape:.2f}%**  
แปลว่าทำนายราคาทองคำรายเดือนผิดพลาดเฉลี่ยเพียง **{holt_mape:.2f}%** จากราคาจริง

**2. ทั้ง 3 วิธีมีผลใกล้เคียงกันมาก** (MAPE ห่างกันไม่ถึง 1%)  
แสดงว่าวิธีทางสถิติแบบคลาสสิกสามารถทำนายราคาทองคำได้ดีในระดับรายเดือน

**3. เหตุผลที่ Holt ชนะ:**
- ราคาทองคำมี **Strong Uptrend** อย่างชัดเจน → Holt จับ Trend ได้ตรงจุด
- **ไม่มี Seasonal Pattern ที่รุนแรง** → พารามิเตอร์ Seasonal ใน Holt-Winters ไม่ช่วยเพิ่มความแม่นยำ
- **ความเรียบง่ายพอดี** (2 พารามิเตอร์) → ไม่ Overfit ไม่ Underfit

**4. คำแนะนำ:**
- ใช้ **Holt** เป็นโมเดลหลักสำหรับทำนายระยะสั้น (1–6 เดือน)
- ใช้ **ARIMA** เป็นโมเดลสำรองเพื่อ Cross-validate
- ใน Production ควรทำ **Ensemble** (เฉลี่ยผลจาก 2–3 วิธี) เพื่อลด Risk
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("หน้า 8 — เปรียบเทียบ 3 วิธีเด่น (Slide-Ready)")
