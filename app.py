"""
XAU/USD Time Series Analysis — Streamlit Multipage Dashboard
Run: streamlit run app.py
"""
import streamlit as st

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="XAU/USD Time Series Analysis",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# CUSTOM CSS — Gold theme
# ──────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%); }
    .main-header {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem; font-weight: 700; text-align: center; padding: 1.5rem 0;
    }
    .sub-header { color: #9CA3AF; text-align: center; font-size: 1.15rem; margin-bottom: 2rem; }
    .gold-card {
        background: linear-gradient(135deg, rgba(255,215,0,0.07), rgba(255,165,0,0.03));
        border: 1px solid rgba(255,215,0,0.18); border-radius: 14px; padding: 1.5rem; margin: 0.5rem 0;
    }
    .gold-card h3 { margin-top: 0; }
    div[data-testid="stMetric"] {
        background: rgba(255,215,0,0.06); border: 1px solid rgba(255,215,0,0.15);
        border-radius: 10px; padding: 0.8rem;
    }
    .stSidebar > div:first-child { background: linear-gradient(180deg, #0d1117, #161b22); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# LANDING PAGE
# ──────────────────────────────────────────
st.markdown('<div class="main-header">🪙 XAU/USD Time Series Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">โครงงานวิเคราะห์อนุกรมเวลาราคาทองคำ XAU/USD ย้อนหลัง 10 ปี (2016 – 2025)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="gold-card">
        <h3>📊 ข้อมูล</h3>
        <p>ราคาทองคำ XAU/USD รายวัน<br>ย้อนหลัง 10 ปี (2016–2025)<br>2,512 วันทำการ จาก Yahoo Finance</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="gold-card">
        <h3>🔬 การวิเคราะห์</h3>
        <p>เปรียบเทียบ 8 วิธีพยากรณ์อนุกรมเวลา<br>SMA · Holt · Holt-Winters · ARIMA · Prophet<br>Linear · Polynomial · Multiple Regression<br>ประเมินด้วย MAE / RMSE / MAPE</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="gold-card">
        <h3>📈 เครื่องมือ</h3>
        <p>Python + Streamlit<br>Plotly · Apache ECharts · Altair<br>statsmodels · scikit-learn · Prophet</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📄 เลือกหน้าจาก Sidebar ด้านซ้าย")
st.markdown("""
| หน้า | เนื้อหา |
|------|---------|
| **1. ที่มาและความสำคัญ** | ความเป็นมา แหล่งข้อมูล พร้อมกราฟภาพรวมราคาทอง |
| **2. วัตถุประสงค์** | เป้าหมายของการวิเคราะห์ |
| **3. ประโยชน์จากการวิเคราะห์** | คุณค่าที่ได้จากโครงงาน |
| **4. สถิติเบื้องต้น** | Descriptive Statistics · Histogram · Box Plot · Heatmap |
| **5. วิเคราะห์อนุกรมเวลา** | เปรียบเทียบ 8 วิธี พร้อมกราฟแต่ละแบบ |
| **6. สรุปรูปแบบที่ดีที่สุด** | เลือก Best Model จาก MAE / RMSE / MAPE |
| **7. กราฟจริง vs ทำนาย** | เส้นแสดงแนวโน้มข้อมูลจริงเทียบข้อมูลทำนาย |
| **8. เปรียบเทียบ 3 วิธีเด่น** | Holt vs Holt-Winters vs ARIMA — กราฟ + อธิบาย (Slide-Ready) |
| **9. Regression & Overlay** | Linear / Polynomial / Multiple Regression + กราฟซ้อนทุกวิธี |
""")

st.markdown("---")
st.caption("XAU/USD Time Series Analysis Project — วิชาคณิตศาสตร์และสถิติ")
