"""
Page 5: วิเคราะห์อนุกรมเวลาโดยเปรียบเทียบอย่างน้อย 4 วิธี
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
from helpers import inject_css, get_data, get_models, decompose_series

inject_css()
df, monthly = get_data()

st.markdown('<div class="page-title">🔬 5. วิเคราะห์อนุกรมเวลา — เปรียบเทียบ 8 วิธี</div>', unsafe_allow_html=True)
st.markdown("---")

# ──────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────
test_months = st.slider("🔧 จำนวนเดือนทดสอบ (Test Period)", 3, 12, 6,
                         help="จำนวนเดือนสุดท้ายที่ใช้เป็นชุดทดสอบ (Test Set)")
train, test, results, metrics_df, model_params = get_models(test_months)

st.markdown(f"""
<div class="section-box">
<h3>📐 การแบ่งข้อมูล (Train/Test Split)</h3>

- **Train Set:** {len(monthly) - test_months} เดือน ({monthly.index[0].strftime('%b %Y')} – {monthly.index[-test_months-1].strftime('%b %Y')})  
- **Test Set:** {test_months} เดือน ({test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')})  
- ใช้ข้อมูล **รายเดือน** (Monthly Average Price) เนื่องจากข้อมูลรายวันมี Noise สูงเกินไป 
  สำหรับแบบจำลองอนุกรมเวลาเชิงสถิติ
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# DECOMPOSITION
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("## 1️⃣ Time Series Decomposition")
st.markdown("""
<div class="highlight-box">
<b>Additive Decomposition</b> แยกข้อมูลรายเดือนเป็น 3 ส่วน:
<br>$ Y_t = T_t + S_t + R_t $
<br>โดย $T_t$ = Trend, $S_t$ = Seasonal (คาบ 12 เดือน), $R_t$ = Residual
<br><br>
<b>ผลวิเคราะห์:</b>
<ul>
<li><b>Trend:</b> ขาขึ้นชัดเจน โดยเฉพาะตั้งแต่กลางปี 2024</li>
<li><b>Seasonal:</b> มีวัฏจักรรายปีเล็กน้อย แต่ไม่รุนแรง</li>
<li><b>Residual:</b> ส่วนใหญ่อยู่ใกล้ 0 แสดงว่าโมเดลอธิบายได้ดีพอสมควร</li>
</ul>
</div>
""", unsafe_allow_html=True)

decomp = decompose_series(monthly)
fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=["Observed (ค่าจริง)", "Trend (แนวโน้ม)",
                                    "Seasonal (ฤดูกาล)", "Residual (ส่วนที่เหลือ)"],
                    vertical_spacing=0.06)
configs = [
    (monthly["avg_price"], "#FFD700", "lines"),
    (decomp.trend, "#4FC3F7", "lines"),
    (decomp.seasonal, "#81C784", "lines"),
    (decomp.resid, "#E57373", "markers+lines"),
]
for i, (series, color, mode) in enumerate(configs, 1):
    fig.add_trace(go.Scatter(
        x=monthly.index, y=series, mode=mode,
        line=dict(color=color, width=2), marker=dict(size=4, color=color),
        showlegend=False,
    ), row=i, col=1)
fig.update_layout(height=750, template="plotly_dark",
                  title_text="Additive Decomposition (Monthly Average Price, Period=12)")
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════
# 5 METHODS — ONE BY ONE
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("## 2️⃣ เปรียบเทียบ 8 วิธีพยากรณ์")

method_info = {
    "SMA": {
        "full_name": "Simple Moving Average (SMA)",
        "color": "#FF6B6B",
        "description": """
        **Simple Moving Average (SMA)** คำนวณค่าเฉลี่ยของข้อมูล $n$ จุดล่าสุด:
        
        $$ SMA_t = \\frac{1}{n} \\sum_{i=0}^{n-1} x_{t-i} $$
        
        - ใช้ window = 3 เดือน
        - **ข้อดี:** เรียบง่าย เข้าใจง่าย คำนวณเร็ว
        - **ข้อเสีย:** ตอบสนองช้า (Lagging) ไม่จับ Trend หรือ Seasonal ได้ดี
        - **เหมาะกับ:** ข้อมูลที่ค่อนข้าง Stationary หรือเป็น Baseline เปรียบเทียบ
        """,
        "metric_key": "SMA",
    },
    "Holt": {
        "full_name": "Holt's Double Exponential Smoothing",
        "color": "#FFEAA7",
        "description": """
        **Holt's Method** (Double Exponential Smoothing) เพิ่มองค์ประกอบ Trend เข้ามา:
        
        $$ \\hat{y}_{t+h} = l_t + h \\cdot b_t $$
        
        โดย $l_t$ = Level, $b_t$ = Trend
        
        - **ข้อดี:** จับ Trend ได้ดีกว่า SMA
        - **ข้อเสีย:** ไม่มี Seasonal Component
        - **เหมาะกับ:** ข้อมูลที่มี Trend ชัดเจนแต่ไม่มีฤดูกาล
        """,
        "metric_key": "Holt",
    },
    "ExpSmoothing": {
        "full_name": "Holt-Winters (Triple Exponential Smoothing)",
        "color": "#4ECDC4",
        "description": """
        **Holt-Winters** เพิ่ม Seasonal Component เข้ามาอีก:
        
        $$ \\hat{y}_{t+h} = l_t + h \\cdot b_t + s_{t+h-m} $$
        
        โดย $s_t$ = Seasonal Component, $m$ = Seasonal Period (12 เดือน)
        
        - ใช้ Additive Model (Trend + Seasonal + Error)
        - **ข้อดี:** จับทั้ง Trend และ Seasonal ได้
        - **ข้อเสีย:** ต้องมีข้อมูลอย่างน้อย 2 รอบ Seasonal (24 เดือน)
        - **เหมาะกับ:** ข้อมูลที่มีทั้ง Trend และ Seasonal Pattern
        """,
        "metric_key": "ExpSmoothing",
    },
    "ARIMA": {
        "full_name": "ARIMA (AutoRegressive Integrated Moving Average)",
        "color": "#45B7D1",
        "description": """
        **ARIMA(p, d, q)** ใช้ ARIMA(2, 1, 2):
        
        $$ (1 - \\phi_1 B - \\phi_2 B^2)(1-B)y_t = (1 + \\theta_1 B + \\theta_2 B^2)\\epsilon_t $$
        
        - $p=2$ (AR terms), $d=1$ (Differencing), $q=2$ (MA terms)
        - **ข้อดี:** ยืดหยุ่นสูง จับ Autocorrelation ได้ดี
        - **ข้อเสีย:** ต้อง Tune Parameters (p, d, q), 
          อาจไม่เหมาะกับข้อมูลที่มี Structural Break
        - **เหมาะกับ:** ข้อมูล Stationary หรือทำ Differencing ให้ Stationary ได้
        """,
        "metric_key": "ARIMA",
    },
    "Prophet": {
        "full_name": "Prophet (Facebook / Meta)",
        "color": "#96CEB4",
        "description": """
        **Prophet** ใช้ Additive Model:
        
        $$ y(t) = g(t) + s(t) + h(t) + \\epsilon_t $$
        
        - $g(t)$ = Piecewise Linear / Logistic Growth
        - $s(t)$ = Fourier Series สำหรับ Seasonality
        - $h(t)$ = Holiday Effects
        
        - **ข้อดี:** ใช้งานง่าย รองรับ Missing Data และ Outliers ได้ดี
        - **ข้อเสีย:** อาจ Overfit ถ้า Seasonal ไม่ชัด
        - **เหมาะกับ:** ข้อมูลที่มี Seasonal Pattern ชัดเจน หรือมี Holiday Effects
        """,
        "metric_key": "Prophet",
    },
    "LinearReg": {
        "full_name": "Linear Regression",
        "color": "#E8A0BF",
        "description": """
        **Linear Regression** ใช้สมการเส้นตรงเพื่อจับแนวโน้มของราคา:
        
        $$ \\hat{y}_t = \\beta_0 + \\beta_1 \\cdot t $$
        
        - $\\beta_0$ = Intercept (จุดตัดแกน y)
        - $\\beta_1$ = Slope (ความชันของแนวโน้ม)
        - $t$ = Time Index (ลำดับเดือน)
        
        - **ข้อดี:** เรียบง่ายมาก ตีความง่าย Baseline ที่ดี
        - **ข้อเสีย:** จับได้แค่แนวโน้มเชิงเส้นเท่านั้น
        - **เหมาะกับ:** ข้อมูลที่มีแนวโน้มเป็นเส้นตรง
        """,
        "metric_key": "LinearReg",
    },
    "PolyReg": {
        "full_name": "Polynomial Regression (Degree 2)",
        "color": "#B983FF",
        "description": """
        **Polynomial Regression** ขยายจาก Linear ด้วย Term กำลังสอง:
        
        $$ \\hat{y}_t = \\beta_0 + \\beta_1 \\cdot t + \\beta_2 \\cdot t^2 $$
        
        - Degree = 2 (Quadratic) เพื่อจับแนวโน้มโค้ง
        - $\\beta_2 > 0$ → โค้งเปิดขึ้น (ราคาเร่งตัว)
        - $\\beta_2 < 0$ → โค้งเปิดลง (ราคาชะลอตัว)
        
        - **ข้อดี:** จับ Non-linear Trend ได้ดีกว่า Linear
        - **ข้อเสีย:** อาจ Extrapolate ผิดพลาดถ้า Degree สูงเกินไป
        - **เหมาะกับ:** ข้อมูลที่มีแนวโน้มโค้งชัดเจน
        """,
        "metric_key": "PolyReg",
    },
    "MultiReg": {
        "full_name": "Multiple Regression",
        "color": "#94D2BD",
        "description": """
        **Multiple Regression** ใช้หลาย Features ร่วมกัน:
        
        $$ \\hat{y}_t = \\beta_0 + \\beta_1 t + \\beta_2 t^2 + \\beta_3 \\sin\\!\\left(\\frac{2\\pi m}{12}\\right) + \\beta_4 \\cos\\!\\left(\\frac{2\\pi m}{12}\\right) $$
        
        - $t$ = Time Index, $t^2$ = Quadratic Trend
        - $\\sin, \\cos$ = Cyclical Encoding ของเดือน (Seasonality)
        
        - **ข้อดี:** จับทั้ง Non-linear Trend และ Seasonal Pattern
        - **ข้อเสีย:** ต้องเลือก Features อย่างเหมาะสม
        - **เหมาะกับ:** ข้อมูลที่มีทั้ง Trend และ Seasonal
        """,
        "metric_key": "MultiReg",
    },
}


# ── Helper: render fitted equation with actual parameter values ──
def render_fitted_equation(method_key, mp_dict):
    """Show the fitted equation and parameter values for a method."""
    if method_key not in mp_dict:
        return
    mp = mp_dict[method_key]

    if method_key == "SMA":
        vals = mp["last_values"]
        st.markdown("""<div class="section-box"><h4>📐 Fitted Parameters — SMA(3)</h4></div>""", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Window (n) | **3 เดือน** |
| x(t−1) | **{vals[2]:,.2f}** |
| x(t−2) | **{vals[1]:,.2f}** |
| x(t−3) | **{vals[0]:,.2f}** |
""")
        st.markdown("**การทำนายเดือนแรก:**")
        st.latex(rf"SMA = \frac{{{vals[2]:,.2f} + {vals[1]:,.2f} + {vals[0]:,.2f}}}{{3}} = {mp['first_pred']:,.2f}")

    elif method_key == "Holt":
        a, b = mp["alpha"], mp["beta"]
        a_inv = round(1 - a, 4)
        b_inv = round(1 - b, 4)
        st.markdown("""<div class="section-box"><h4>📐 Fitted Parameters — Holt's Double Exponential Smoothing</h4></div>""", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value | คำอธิบาย |
|-----------|-------|---------|
| α (smoothing level) | **{a}** | น้ำหนักสำหรับ Level |
| β (smoothing trend) | **{b}** | น้ำหนักสำหรับ Trend |
| l₀ (initial level) | **{mp['initial_level']:,.2f}** | ค่าเริ่มต้น Level |
| b₀ (initial trend) | **{mp['initial_trend']:,.2f}** | ค่าเริ่มต้น Trend |
| AIC | **{mp['aic']:,.2f}** | Akaike Information Criterion |
| BIC | **{mp['bic']:,.2f}** | Bayesian Information Criterion |
| SSE | **{mp['sse']:,.2f}** | Sum of Squared Errors |
""")
        st.markdown("**สมการ Fitted:**")
        st.latex(rf"l_t = {a} \cdot y_t + {a_inv} \cdot (l_{{t-1}} + b_{{t-1}})")
        st.latex(rf"b_t = {b} \cdot (l_t - l_{{t-1}}) + {b_inv} \cdot b_{{t-1}}")
        st.latex(r"\hat{y}_{t+h} = l_t + h \cdot b_t")

    elif method_key == "ExpSmoothing":
        a, b, g = mp["alpha"], mp["beta"], mp["gamma"]
        a_inv = round(1 - a, 4)
        b_inv = round(1 - b, 4)
        g_inv = round(1 - g, 4)
        st.markdown("""<div class="section-box"><h4>📐 Fitted Parameters — Holt-Winters (Additive)</h4></div>""", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value | คำอธิบาย |
|-----------|-------|---------|
| α (smoothing level) | **{a}** | น้ำหนักสำหรับ Level |
| β (smoothing trend) | **{b}** | น้ำหนักสำหรับ Trend |
| γ (smoothing seasonal) | **{g}** | น้ำหนักสำหรับ Seasonal |
| m (seasonal period) | **12 เดือน** | ความยาวรอบฤดูกาล |
| AIC | **{mp['aic']:,.2f}** | Akaike Information Criterion |
| BIC | **{mp['bic']:,.2f}** | Bayesian Information Criterion |
| SSE | **{mp['sse']:,.2f}** | Sum of Squared Errors |
""")
        st.markdown("**สมการ Fitted:**")
        st.latex(rf"l_t = {a}(y_t - s_{{t-12}}) + {a_inv}(l_{{t-1}} + b_{{t-1}})")
        st.latex(rf"b_t = {b}(l_t - l_{{t-1}}) + {b_inv} \cdot b_{{t-1}}")
        st.latex(rf"s_t = {g}(y_t - l_t) + {g_inv} \cdot s_{{t-12}}")
        st.latex(r"\hat{y}_{t+h} = l_t + h \cdot b_t + s_{t+h-12}")

    elif method_key == "ARIMA":
        phi = mp["ar_params"]
        theta = mp["ma_params"]
        st.markdown("""<div class="section-box"><h4>📐 Fitted Parameters — ARIMA(2,1,2)</h4></div>""", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value | คำอธิบาย |
|-----------|-------|---------|
| φ₁ (AR.L1) | **{phi[0]}** | สัมประสิทธิ์ Autoregressive ลำดับ 1 |
| φ₂ (AR.L2) | **{phi[1]}** | สัมประสิทธิ์ Autoregressive ลำดับ 2 |
| θ₁ (MA.L1) | **{theta[0]}** | สัมประสิทธิ์ Moving Average ลำดับ 1 |
| θ₂ (MA.L2) | **{theta[1]}** | สัมประสิทธิ์ Moving Average ลำดับ 2 |
| σ² | **{mp['sigma2']:,.2f}** | Variance ของ Residual |
| AIC | **{mp['aic']:,.2f}** | Akaike Information Criterion |
| BIC | **{mp['bic']:,.2f}** | Bayesian Information Criterion |
""")
        st.markdown("**สมการ Fitted (Lag Operator):**")
        st.latex(rf"(1 - ({phi[0]})B - ({phi[1]})B^2)(1-B)y_t = (1 + ({theta[0]})B + ({theta[1]})B^2)\varepsilon_t")
        st.markdown("**Expanded (ใส่ค่าสัมประสิทธิ์แล้ว):**")
        st.latex(rf"\Delta y_t = {phi[0]} \cdot \Delta y_{{t-1}} + {phi[1]} \cdot \Delta y_{{t-2}} + {theta[0]} \cdot \varepsilon_{{t-1}} + ({theta[1]}) \cdot \varepsilon_{{t-2}} + \varepsilon_t")
        st.markdown(r"โดย $\Delta y_t = y_t - y_{t-1}$ (First Differencing, $d\!=\!1$)")

    elif method_key == "Prophet":
        gr = mp.get("growth_rate", "N/A")
        off = mp.get("offset", "N/A")
        nc = mp.get("n_changepoints", 0)
        st.markdown("""<div class="section-box"><h4>📐 Fitted Parameters — Prophet</h4></div>""", unsafe_allow_html=True)
        st.markdown(f"""
| Parameter | Value | คำอธิบาย |
|-----------|-------|---------|
| Growth | **Linear** | Piecewise Linear Growth |
| k (growth rate) | **{gr}** | อัตราการเติบโต (ความชัน Trend) |
| m (offset) | **{off}** | ค่า offset ของ Trend |
| Changepoints | **{nc} จุด** | จุดเปลี่ยน Trend |
| Yearly Seasonality | **True** | Fourier Series (order=10) |
""")
        st.markdown("**สมการหลัก:**")
        st.latex(r"y(t) = g(t) + s(t) + \varepsilon_t")
        st.markdown(f"**Growth:** $g(t) = k \\cdot t + m$ โดย $k = {gr}$")
        st.markdown("**Seasonality (Fourier Series):**")
        st.latex(r"s(t) = \sum_{n=1}^{N} \left( a_n \cos\frac{2\pi nt}{365.25} + b_n \sin\frac{2\pi nt}{365.25} \right)")


for method_key, pred_series in results.items():
    info = method_info.get(method_key, {})
    color = info.get("color", "#FFF")
    full_name = info.get("full_name", method_key)

    st.markdown(f"### 📌 {full_name}")
    st.markdown(f"""<div class="section-box">{info.get('description', '')}</div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        # ECharts line chart for each method
        train_dates = train.index.strftime("%Y-%m-%d").tolist()
        test_dates = test.index.strftime("%Y-%m-%d").tolist()
        all_dates = train_dates + test_dates

        train_prices = [round(v, 2) for v in train["avg_price"].values]
        test_actual = [None] * len(train_dates) + [round(v, 2) for v in test["avg_price"].values]
        test_pred = [None] * len(train_dates) + [round(v, 2) for v in pred_series.values]

        method_chart_opt = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Train", "Actual (Test)", f"Predicted ({method_key})"],
                       "textStyle": {"color": "#CCC"}},
            "xAxis": {"type": "category", "data": all_dates, "boundaryGap": False},
            "yAxis": {"type": "value", "axisLabel": {"formatter": "${value}"}},
            "series": [
                {"name": "Train", "type": "line", "data": train_prices + [None]*len(test_dates),
                 "lineStyle": {"color": "#555"}, "showSymbol": False},
                {"name": "Actual (Test)", "type": "line", "data": test_actual,
                 "lineStyle": {"color": "#FFD700", "width": 3},
                 "symbol": "circle", "symbolSize": 8, "itemStyle": {"color": "#FFD700"}},
                {"name": f"Predicted ({method_key})", "type": "line", "data": test_pred,
                 "lineStyle": {"color": color, "width": 3, "type": "dashed"},
                 "symbol": "diamond", "symbolSize": 8, "itemStyle": {"color": color}},
            ],
            "backgroundColor": "transparent",
        }
        st_echarts(options=method_chart_opt, height="380px")

    with col2:
        # Find matching metric row
        match_map = {
            "SMA": "SMA", "ExpSmoothing": "Holt-Winters",
            "ARIMA": "ARIMA", "Prophet": "Prophet", "Holt": "Holt",
            "LinearReg": "Linear Regression", "PolyReg": "Polynomial",
            "MultiReg": "Multiple Regression",
        }
        match_str = match_map.get(method_key, method_key)
        row = metrics_df[metrics_df["method"].str.contains(match_str)]
        if not row.empty:
            r = row.iloc[0]
            st.metric("MAE", f"${r['MAE']:,.2f}")
            st.metric("RMSE", f"${r['RMSE']:,.2f}")
            st.metric("MAPE", f"{r['MAPE (%)']:.2f}%")

            # Mini interpretation
            if r["MAPE (%)"] < 7:
                st.success("✅ ค่าความผิดพลาดต่ำ")
            elif r["MAPE (%)"] < 10:
                st.warning("⚠️ ค่าความผิดพลาดปานกลาง")
            else:
                st.error("❌ ค่าความผิดพลาดสูง")

    # ── Fitted Equation & Parameters ──
    render_fitted_equation(method_key, model_params)

    st.markdown("---")

# ═══════════════════════════════════════
# METRICS COMPARISON TABLE
# ═══════════════════════════════════════
st.markdown("## 3️⃣ ตารางเปรียบเทียบ Metrics ทุกวิธี")
st.markdown("""
<div class="highlight-box">
<b>เกณฑ์การประเมิน:</b>
<ul>
<li><b>MAE</b> (Mean Absolute Error) — ค่าความผิดพลาดเฉลี่ยสัมบูรณ์ (หน่วย USD) ยิ่งน้อยยิ่งดี</li>
<li><b>RMSE</b> (Root Mean Squared Error) — ลงโทษ Error ใหญ่ๆ มากกว่า MAE ยิ่งน้อยยิ่งดี</li>
<li><b>MAPE</b> (Mean Absolute Percentage Error) — เปอร์เซ็นต์ความผิดพลาดเฉลี่ย ยิ่งน้อยยิ่งดี</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.dataframe(
    metrics_df.style
    .highlight_min(subset=["MAE", "RMSE", "MAPE (%)"], color="rgba(45,106,79,0.4)")
    .highlight_max(subset=["MAE", "RMSE", "MAPE (%)"], color="rgba(255,107,107,0.3)")
    .format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE (%)": "{:.2f}%"}),
    use_container_width=True, hide_index=True,
)

# ═══════════════════════════════════════
# BAR CHART: MAPE comparison via ECharts
# ═══════════════════════════════════════
st.markdown("---")
st.markdown("## 4️⃣ กราฟเปรียบเทียบ MAPE")

bar_colors = ["#81C784", "#A5D6A7", "#C8E6C9", "#FFD54F", "#FFB74D", "#FFCC80", "#FF8A65", "#E57373"]
mape_data = []
for i, (_, row) in enumerate(metrics_df.iterrows()):
    mape_data.append({
        "value": row["MAPE (%)"],
        "itemStyle": {"color": bar_colors[i] if i < len(bar_colors) else "#888"},
    })

mape_bar_opt = {
    "tooltip": {"trigger": "axis", "formatter": "{b}: {c}%"},
    "xAxis": {"type": "category", "data": metrics_df["method"].tolist(),
              "axisLabel": {"rotate": 15}},
    "yAxis": {"type": "value", "name": "MAPE (%)", "axisLabel": {"formatter": "{value}%"}},
    "series": [{
        "type": "bar",
        "data": mape_data,
        "label": {"show": True, "position": "top", "formatter": "{c}%"},
        "barWidth": "50%",
        "itemStyle": {"borderRadius": [8, 8, 0, 0]},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=mape_bar_opt, height="400px")

# ═══════════════════════════════════════
# GROUPED BAR: MAE & RMSE
# ═══════════════════════════════════════
st.markdown("## 5️⃣ MAE & RMSE Comparison")

grouped_bar_opt = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["MAE", "RMSE"], "textStyle": {"color": "#CCC"}},
    "xAxis": {"type": "category", "data": metrics_df["method"].tolist()},
    "yAxis": {"type": "value", "name": "USD", "axisLabel": {"formatter": "${value}"}},
    "series": [
        {"name": "MAE", "type": "bar", "data": metrics_df["MAE"].tolist(),
         "itemStyle": {"color": "#4ECDC4", "borderRadius": [6, 6, 0, 0]}},
        {"name": "RMSE", "type": "bar", "data": metrics_df["RMSE"].tolist(),
         "itemStyle": {"color": "#FF6B6B", "borderRadius": [6, 6, 0, 0]}},
    ],
    "backgroundColor": "transparent",
}
st_echarts(options=grouped_bar_opt, height="400px")

st.markdown("---")
st.caption("หน้า 5/9 — วิเคราะห์อนุกรมเวลาเปรียบเทียบ 8 วิธี")
