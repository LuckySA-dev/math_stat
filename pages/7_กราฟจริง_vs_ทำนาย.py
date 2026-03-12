"""
Page 7: สร้างกราฟเส้นแสดงแนวโน้มข้อมูลจริงกับข้อมูลทำนาย
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
from helpers import inject_css, get_data, get_models

inject_css()
df, monthly = get_data()

st.markdown('<div class="page-title">📉 7. กราฟเส้นแสดงแนวโน้ม: ข้อมูลจริง vs ข้อมูลทำนาย</div>', unsafe_allow_html=True)
st.markdown("---")

test_months = st.slider("🔧 จำนวนเดือนทดสอบ", 3, 12, 6, key="p7_slider")
train, test, results, metrics_df, model_params = get_models(test_months)

# ═══════════════════════════════════════
# COMBINED CHART: ALL METHODS
# ═══════════════════════════════════════
st.markdown("## 📊 ทุกวิธีรวมในกราฟเดียว")
st.markdown("""
<div class="highlight-box">
กราฟด้านล่างแสดง <b>ราคาจริง (Actual)</b> เทียบกับ <b>ราคาทำนาย (Predicted)</b> 
จากทั้ง 8 วิธี ในช่วง Test Period 
— เส้นทึบสีทอง = ราคาจริง, เส้นประ = ราคาทำนายจากแต่ละวิธี
</div>
""", unsafe_allow_html=True)

# Build ECharts combined chart
all_dates = monthly.index.strftime("%Y-%m-%d").tolist()
train_end_idx = len(train)

# Full actual line
actual_full = [round(v, 2) for v in monthly["avg_price"].values]

# Build prediction series (None for train period, values for test period)
method_colors = {
    "SMA": "#FF6B6B", "Holt": "#FFEAA7", "ExpSmoothing": "#4ECDC4",
    "ARIMA": "#45B7D1", "Prophet": "#96CEB4",
    "LinearReg": "#E8A0BF", "PolyReg": "#B983FF", "MultiReg": "#94D2BD",
}

METHOD_NAMES = {
    "SMA": "SMA (3-month)", "Holt": "Holt", "ExpSmoothing": "Holt-Winters",
    "ARIMA": "ARIMA(2,1,2)", "Prophet": "Prophet",
    "LinearReg": "Linear Reg", "PolyReg": "Poly Reg", "MultiReg": "Multi Reg",
}
METHOD_DASH = {
    "SMA": "dashed", "Holt": "dashed", "ExpSmoothing": "dotted",
    "ARIMA": "dashed", "Prophet": "dotted",
    "LinearReg": "dashed", "PolyReg": "dotted", "MultiReg": "dashed",
}
METHOD_SYMBOL = {
    "SMA": "circle", "Holt": "diamond", "ExpSmoothing": "triangle",
    "ARIMA": "rect", "Prophet": "roundRect",
    "LinearReg": "pin", "PolyReg": "arrow", "MultiReg": "circle",
}

echarts_series = [
    {
        "name": "Actual (ราคาจริง)",
        "type": "line",
        "data": actual_full,
        "lineStyle": {"color": "#FFD700", "width": 3},
        "showSymbol": False,
        "emphasis": {"lineStyle": {"width": 4}},
        "z": 10,
    }
]

for method_key, pred in results.items():
    pred_data = [None] * train_end_idx + [round(v, 2) for v in pred.values]
    color = method_colors.get(method_key, "#888")
    echarts_series.append({
        "name": METHOD_NAMES.get(method_key, method_key),
        "type": "line",
        "data": pred_data,
        "lineStyle": {"color": color, "width": 2.5, "type": METHOD_DASH.get(method_key, "dashed")},
        "symbol": METHOD_DASH.get(method_key, "diamond"),
        "symbolSize": 8,
        "itemStyle": {"color": color},
    })

# Compute nice y-axis range for the zoomed-in view
_zoom_prices = list(monthly["avg_price"].values[-test_months-12:])
for _pred in results.values():
    _zoom_prices.extend(_pred.values)
_y_min = int(min(_zoom_prices) * 0.92 // 100) * 100
_y_max = int(max(_zoom_prices) * 1.05 // 100 + 1) * 100

combined_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {
        "data": ["Actual (ราคาจริง)"] + [METHOD_NAMES.get(k, k) for k in results.keys()],
        "textStyle": {"color": "#CCC", "fontSize": 11}, "bottom": 0, "type": "scroll",
    },
    "grid": {"top": 40, "bottom": 70, "left": 65, "right": 25},
    "dataZoom": [
        {"type": "inside", "start": max(0, 100 - (test_months + 18) * 100 / len(monthly)), "end": 100},
        {"start": max(0, 100 - (test_months + 18) * 100 / len(monthly)), "end": 100, "bottom": 30},
    ],
    "xAxis": {"type": "category", "data": all_dates, "boundaryGap": False},
    "yAxis": {"type": "value", "min": _y_min, "max": _y_max, "axisLabel": {"formatter": "${value}"}},
    "series": echarts_series,
    "backgroundColor": "transparent",
}
st_echarts(options=combined_opt, height="550px")

st.markdown("---")

# ═══════════════════════════════════════
# INDIVIDUAL METHOD SELECTOR
# ═══════════════════════════════════════
st.markdown("## 🔎 เลือกดูรายวิธี")
selected = st.selectbox("เลือกวิธีพยากรณ์", list(results.keys()), key="method_select")
pred = results[selected]
error = test["avg_price"].values - pred.values

col1, col2 = st.columns([2, 1])

sel_color = method_colors.get(selected, "#4ECDC4")
sel_name = METHOD_NAMES.get(selected, selected)

with col1:
    st.markdown(f"### {sel_name}: Actual vs Predicted")

    # Plotly — zoomed into last N+6 months for clarity
    zoom_start = max(0, len(train) - 6)
    zoom_monthly = monthly.iloc[zoom_start:]

    fig = go.Figure()

    # Actual (zoomed)
    fig.add_trace(go.Scatter(
        x=zoom_monthly.index, y=zoom_monthly["avg_price"],
        mode="lines+markers", name="Actual (ราคาจริง)",
        line=dict(color="#FFD700", width=3),
        marker=dict(size=6, color="#FFD700"),
    ))

    # Prediction
    fig.add_trace(go.Scatter(
        x=test.index, y=pred.values,
        mode="lines+markers", name=f"Predicted ({sel_name})",
        line=dict(color=sel_color, width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond", color=sel_color,
                    line=dict(width=1.5, color="white")),
    ))

    # Simple confidence band (±1 std of error)
    err_std = np.std(error)
    hex_rgb = sel_color.lstrip("#")
    r, g, b = int(hex_rgb[:2], 16), int(hex_rgb[2:4], 16), int(hex_rgb[4:6], 16)
    fig.add_trace(go.Scatter(
        x=list(test.index) + list(test.index[::-1]),
        y=list(pred.values + err_std) + list((pred.values - err_std)[::-1]),
        fill="toself", fillcolor=f"rgba({r},{g},{b},0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1σ Confidence Band",
        showlegend=True,
    ))

    # Train/test split line
    fig.add_shape(
        type="line",
        x0=test.index[0], x1=test.index[0],
        y0=0, y1=1, yref="paper",
        line=dict(dash="dot", color="rgba(255,255,255,0.4)", width=1.5),
    )
    fig.add_annotation(
        x=test.index[0], y=1.04, yref="paper",
        text="← Train | Test →",
        font=dict(color="#AAA", size=12),
        showarrow=False,
    )

    # Compute tight y-range
    all_y = list(zoom_monthly["avg_price"].values) + list(pred.values)
    y_pad = (max(all_y) - min(all_y)) * 0.12
    fig.update_layout(
        template="plotly_dark", height=500,
        title=dict(text=f"{sel_name}: Actual vs Predicted", font=dict(size=16)),
        xaxis_title="Date", yaxis_title="Price (USD)",
        yaxis=dict(range=[min(all_y) - y_pad, max(all_y) + y_pad]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📋 ตารางเปรียบเทียบ")
    compare_df = pd.DataFrame({
        "เดือน": test.index.strftime("%b %Y"),
        "ราคาจริง ($)": test["avg_price"].round(2).values,
        "ทำนาย ($)": pred.round(2).values,
        "Error ($)": error.round(2),
        "Error (%)": ((error / test["avg_price"].values) * 100).round(2),
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # Summary stats
    st.markdown("---")
    st.markdown("**สรุป Error:**")
    st.markdown(f"- Mean Error: **${np.mean(error):+,.2f}**")
    st.markdown(f"- Std Error: **${np.std(error):,.2f}**")
    st.markdown(f"- Max |Error|: **${np.max(np.abs(error)):,.2f}**")

st.markdown("---")

# ═══════════════════════════════════════
# WATERFALL: Monthly Errors
# ═══════════════════════════════════════
st.markdown(f"## 📊 Waterfall: Error รายเดือน ({selected})")
st.markdown("""
<div class="highlight-box">
กราฟ Waterfall แสดงค่า Error (ราคาจริง - ราคาทำนาย) ของแต่ละเดือน  
<b>สีเขียว</b> = ทำนายต่ำกว่าจริง (Under-predict) — <b>สีแดง</b> = ทำนายสูงกว่าจริง (Over-predict)
</div>
""", unsafe_allow_html=True)

waterfall_opt = {
    "tooltip": {"trigger": "axis", "formatter": "{b}: ${c}"},
    "xAxis": {"type": "category", "data": test.index.strftime("%b %Y").tolist()},
    "yAxis": {"type": "value", "name": "Error (USD)", "axisLabel": {"formatter": "${value}"}},
    "series": [{
        "type": "bar",
        "data": [
            {
                "value": round(e, 2),
                "itemStyle": {
                    "color": "#81C784" if e > 0 else "#E57373",
                    "borderRadius": [6, 6, 0, 0] if e > 0 else [0, 0, 6, 6],
                },
            }
            for e in error
        ],
        "label": {"show": True, "position": "top", "formatter": "${c}",
                  "fontSize": 11},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=waterfall_opt, height="400px")

st.markdown("---")

# ═══════════════════════════════════════
# CANDLESTICK CHART — Enhanced with MA & Bollinger Bands
# ═══════════════════════════════════════
st.markdown("## 🕯️ Monthly Candlestick Chart")
st.markdown("""
<div class="highlight-box">
กราฟแท่งเทียน (Candlestick) แสดงราคาเปิด-ปิด-สูงสุด-ต่ำสุดรายเดือน พร้อม MA5, MA20 และ Bollinger Bands (±2σ)<br>
<b>แท่งเขียว</b> = เดือนที่ราคาปิดสูงกว่าเปิด (ขาขึ้น) &nbsp;
<b>แท่งแดง</b> = เดือนที่ราคาปิดต่ำกว่าเปิด (ขาลง)<br>
<b>เส้นฟ้า</b> = MA5 &nbsp; <b>เส้นเหลือง</b> = MA20 &nbsp;
<b>เส้นประเขียว</b> = Bollinger Bands (±2σ จาก MA20)
</div>
""", unsafe_allow_html=True)

candle_dates = monthly.index.strftime("%Y-%m-%d").tolist()
candle_data = []
for _, row in monthly.iterrows():
    candle_data.append([
        round(row["open"], 2),
        round(row["avg_price"], 2),
        round(row["low"], 2),
        round(row["high"], 2),
    ])

# MAs and Bollinger Bands
candle_ma5 = monthly["avg_price"].rolling(5).mean()
candle_ma20 = monthly["avg_price"].rolling(20).mean()
bb_std = monthly["avg_price"].rolling(20).std()
bb_upper = candle_ma20 + 2 * bb_std
bb_lower = candle_ma20 - 2 * bb_std

candlestick_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {"data": ["K-Line", "MA5", "MA20", "BB Upper", "BB Lower"],
               "textStyle": {"color": "#CCC"}, "bottom": 35},
    "grid": {"bottom": 80},
    "xAxis": {"type": "category", "data": candle_dates, "boundaryGap": True},
    "yAxis": {"type": "value", "min": 900, "axisLabel": {"formatter": "${value}"}},
    "dataZoom": [
        {"type": "inside", "start": 60, "end": 100},
        {"start": 60, "end": 100, "bottom": 10},
    ],
    "series": [
        {
            "name": "K-Line",
            "type": "candlestick",
            "data": candle_data,
            "itemStyle": {
                "color": "#81C784",
                "color0": "#E57373",
                "borderColor": "#4CAF50",
                "borderColor0": "#F44336",
            },
        },
        {
            "name": "MA5",
            "type": "line",
            "data": [round(v, 2) if not np.isnan(v) else None for v in candle_ma5],
            "smooth": True,
            "lineStyle": {"color": "#4ECDC4", "width": 1.5},
            "showSymbol": False,
        },
        {
            "name": "MA20",
            "type": "line",
            "data": [round(v, 2) if not np.isnan(v) else None for v in candle_ma20],
            "smooth": True,
            "lineStyle": {"color": "#FFEAA7", "width": 1.5},
            "showSymbol": False,
        },
        {
            "name": "BB Upper",
            "type": "line",
            "data": [round(v, 2) if not np.isnan(v) else None for v in bb_upper],
            "lineStyle": {"color": "rgba(150,206,180,0.5)", "width": 1, "type": "dashed"},
            "showSymbol": False,
        },
        {
            "name": "BB Lower",
            "type": "line",
            "data": [round(v, 2) if not np.isnan(v) else None for v in bb_lower],
            "lineStyle": {"color": "rgba(150,206,180,0.5)", "width": 1, "type": "dashed"},
            "showSymbol": False,
        },
    ],
    "backgroundColor": "transparent",
}
st_echarts(options=candlestick_opt, height="550px")

# ═══════════════════════════════════════
# MOVING AVERAGE OVERLAY (ECharts)
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 Price + Moving Averages")
st.markdown("""
<div class="highlight-box">
กราฟราคารายวันพร้อมเส้น Moving Average 20 วัน (MA20) และ 50 วัน (MA50)  
เมื่อ MA20 ตัดขึ้นเหนือ MA50 (Golden Cross) มักเป็นสัญญาณขาขึ้น  
เมื่อ MA20 ตัดลงต่ำกว่า MA50 (Death Cross) มักเป็นสัญญาณขาลง
</div>
""", unsafe_allow_html=True)

daily_dates = df.index.strftime("%Y-%m-%d").tolist()
daily_prices = df["price"].tolist()
ma20 = df["price"].rolling(20).mean()
ma50 = df["price"].rolling(50).mean()

ma_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "legend": {"data": ["Price", "MA20", "MA50"], "textStyle": {"color": "#CCC"}},
    "dataZoom": [
        {"type": "inside", "start": 70, "end": 100},
        {"start": 70, "end": 100},
    ],
    "xAxis": {"type": "category", "data": daily_dates, "boundaryGap": False},
    "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
    "series": [
        {"name": "Price", "type": "line", "data": daily_prices,
         "lineStyle": {"color": "#FFD700", "width": 1}, "showSymbol": False,
         "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                  "colorStops": [{"offset": 0, "color": "rgba(255,215,0,0.15)"},
                                                  {"offset": 1, "color": "rgba(255,215,0,0.01)"}]}}},
        {"name": "MA20", "type": "line",
         "data": [round(v, 2) if not np.isnan(v) else None for v in ma20],
         "lineStyle": {"color": "#4ECDC4", "width": 2}, "showSymbol": False},
        {"name": "MA50", "type": "line",
         "data": [round(v, 2) if not np.isnan(v) else None for v in ma50],
         "lineStyle": {"color": "#FF6B6B", "width": 2}, "showSymbol": False},
    ],
    "backgroundColor": "transparent",
}
st_echarts(options=ma_opt, height="500px")

st.markdown("---")
st.caption("หน้า 7/9 — กราฟเส้นแสดงแนวโน้มข้อมูลจริงกับข้อมูลทำนาย")
