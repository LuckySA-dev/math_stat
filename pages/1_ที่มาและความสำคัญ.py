"""
Page 1: ที่มาและความสำคัญของโครงงาน
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
from helpers import inject_css, get_data

inject_css()
df, monthly = get_data()

# ──────────────────────────────────────
# TITLE
# ──────────────────────────────────────
st.markdown('<div class="page-title">📖 1. ที่มาและความสำคัญของโครงงาน</div>', unsafe_allow_html=True)
st.markdown("---")

# ──────────────────────────────────────
# ที่มา
# ──────────────────────────────────────
st.markdown("""
<div class="section-box">
<h3>🏛️ ที่มาของโครงงาน</h3>

**ทองคำ (Gold – XAU/USD)** เป็นสินทรัพย์ที่มีบทบาทสำคัญในระบบการเงินโลกมาอย่างยาวนาน
โดยเฉพาะในฐานะ **สินทรัพย์ปลอดภัย (Safe-Haven Asset)** ที่นักลงทุนเลือกถือครอง
เมื่อเผชิญกับสถานการณ์ที่ไม่แน่นอน เช่น:

- 🦠 วิกฤต COVID-19 (2020-2021) ที่ทำให้เศรษฐกิจโลกชะลอตัว
- 📈 ภาวะเงินเฟ้อสูง (2022-2023) จากนโยบายกระตุ้นเศรษฐกิจหลังโควิด
- ⚔️ ความขัดแย้งทางภูมิรัฐศาสตร์ (สงครามรัสเซีย-ยูเครน, ตะวันออกกลาง)
- 💰 นโยบายดอกเบี้ยของธนาคารกลางสหรัฐ (Federal Reserve)

จากปัจจัยเหล่านี้ ราคาทองคำในช่วง **10 ปี (2016-2025)** มีความผันผวนและแนวโน้มที่น่าสนใจ
ทำให้เป็นชุดข้อมูลที่เหมาะสำหรับการศึกษาด้วย **การวิเคราะห์อนุกรมเวลา (Time Series Analysis)**
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# KEY METRICS
# ──────────────────────────────────────
st.markdown("### 📊 ภาพรวมข้อมูล")
start_price = df["price"].iloc[0]
end_price = df["price"].iloc[-1]
change_pct = ((end_price - start_price) / start_price) * 100

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("ราคาเริ่มต้น", f"${start_price:,.2f}", f"Jan 2016")
with c2:
    st.metric("ราคาล่าสุด", f"${end_price:,.2f}", f"Dec 2025")
with c3:
    st.metric("การเปลี่ยนแปลง", f"{change_pct:+.1f}%", f"${end_price - start_price:+,.0f}")
with c4:
    st.metric("ราคาสูงสุด", f"${df['price'].max():,.2f}")
with c5:
    st.metric("ราคาต่ำสุด", f"${df['price'].min():,.2f}")

# ──────────────────────────────────────
# ECHARTS: price area chart (beautiful)
# ──────────────────────────────────────
st.markdown("### 📈 กราฟราคาทองคำ XAU/USD (2016–2025)")
st.markdown("""
<div class="highlight-box">
กราฟด้านล่างแสดงราคาปิดรายวัน (Close Price) ของทองคำ XAU/USD ตลอดช่วง 10 ปี 
สังเกตได้ว่าราคามีแนวโน้มขาขึ้น (Uptrend) อย่างชัดเจน โดยเฉพาะช่วงปลายปี 2024 ถึง 2025 
ที่ราคาทะลุ $4,000 เป็นครั้งแรก
</div>
""", unsafe_allow_html=True)

dates = df.index.strftime("%Y-%m-%d").tolist()
prices = df["price"].tolist()

echarts_opt = {
    "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    "toolbox": {"feature": {"dataZoom": {"yAxisIndex": "none"}, "restore": {}, "saveAsImage": {}}},
    "xAxis": {"type": "category", "data": dates, "boundaryGap": False},
    "yAxis": {"type": "value", "min": 900, "axisLabel": {"formatter": "${value}"}},
    "dataZoom": [
        {"type": "inside", "start": 0, "end": 100},
        {"start": 0, "end": 100},
    ],
    "series": [{
        "name": "Close Price",
        "type": "line",
        "data": prices,
        "smooth": True,
        "showSymbol": False,
        "lineStyle": {"color": "#FFD700", "width": 2},
        "areaStyle": {
            "color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": "rgba(255,215,0,0.4)"},
                    {"offset": 1, "color": "rgba(255,215,0,0.02)"},
                ],
            }
        },
        "markPoint": {
            "data": [
                {"type": "max", "name": "สูงสุด"},
                {"type": "min", "name": "ต่ำสุด"},
            ]
        },
        "markLine": {
            "data": [{"type": "average", "name": "เฉลี่ย"}],
            "lineStyle": {"color": "#4ECDC4", "type": "dashed"},
        },
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=echarts_opt, height="500px")

# ──────────────────────────────────────
# ความสำคัญ
# ──────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="section-box">
<h3>💡 ความสำคัญของโครงงาน</h3>

1. **การพยากรณ์ราคาทองคำ** ช่วยให้นักลงทุนสามารถวางแผนและตัดสินใจซื้อ-ขายบนพื้นฐานข้อมูลเชิงสถิติ 
   แทนที่จะตัดสินใจตามอารมณ์หรือข่าวสาร

2. **การวิเคราะห์อนุกรมเวลา** (Time Series Analysis) เป็นเครื่องมือทางสถิติที่ออกแบบมา
   โดยเฉพาะสำหรับข้อมูลที่เรียงตามลำดับเวลา สามารถแยกองค์ประกอบ (Trend, Seasonal, Residual) 
   ที่ซ่อนอยู่ในข้อมูลได้

3. **การเปรียบเทียบหลายวิธี** ช่วยให้เห็นจุดแข็ง-จุดอ่อนของแต่ละแบบจำลอง 
   และเลือกวิธีที่เหมาะสมที่สุดสำหรับข้อมูลชุดนี้
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# แหล่งข้อมูล
# ──────────────────────────────────────
st.markdown("""
<div class="section-box">
<h3>📂 แหล่งข้อมูลและอ้างอิง</h3>

| รายการ | รายละเอียด |
|--------|-----------|
| **Dataset** | XAU/USD Historical Data (Daily) |
| **แหล่งที่มา** | [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF/) (GC=F Gold Futures) |
| **ช่วงเวลา** | 4 มกราคม 2016 – 30 ธันวาคม 2025 |
| **จำนวนข้อมูล** | 2,512 วันทำการ (120 เดือน) |
| **ตัวแปร** | Date, Close Price, Open, High, Low, Change% |
| **ไฟล์** | `datasets/xauusd_2016-2025.csv` |
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# YEARLY BAR via ECharts
# ──────────────────────────────────────
st.markdown("### 📊 ราคาเฉลี่ยรายปี")
yearly_avg = df.groupby(df.index.year)["price"].mean()
bar_opt = {
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": [str(y) for y in yearly_avg.index]},
    "yAxis": {"type": "value", "min": 900, "axisLabel": {"formatter": "${value}"}},
    "series": [{
        "type": "bar",
        "data": [round(v, 2) for v in yearly_avg.values],
        "itemStyle": {
            "color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": "#FFD700"},
                    {"offset": 1, "color": "#FF8C00"},
                ],
            },
            "borderRadius": [8, 8, 0, 0],
        },
        "label": {"show": True, "position": "top", "formatter": "${c}"},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=bar_opt, height="400px")

st.markdown("---")
st.caption("หน้า 1/7 — ที่มาและความสำคัญของโครงงาน")
