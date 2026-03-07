"""
Page 3: ประโยชน์จากการวิเคราะห์
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from streamlit_echarts import st_echarts
from helpers import inject_css

inject_css()

st.markdown('<div class="page-title">💡 3. ประโยชน์จากผลการวิเคราะห์</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="section-box">
<h3>คุณค่าที่ได้จากโครงงานนี้</h3>

การวิเคราะห์อนุกรมเวลาของราคาทองคำ XAU/USD ด้วยวิธีทางสถิติหลายวิธี 
ให้ประโยชน์ในหลายมิติ ทั้งด้านการลงทุน ด้านวิชาการ และด้านธุรกิจ
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# 3 กลุ่มประโยชน์
# ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="section-box">
    <h4>📈 ด้านการลงทุน (Investment)</h4>
    
    - **พยากรณ์ราคาระยะสั้น** — นักลงทุนสามารถใช้แบบจำลองที่เหมาะสมที่สุด 
      เป็นข้อมูลประกอบการตัดสินใจซื้อ-ขายทองคำ
    - **เข้าใจรูปแบบราคา** — เห็นแนวโน้ม (Trend) และรูปแบบตามฤดูกาล (Seasonal) 
      ที่ซ่อนอยู่ในข้อมูลรายวัน
    - **ประเมินความเสี่ยง** — สถิติเบื้องต้น เช่น Standard Deviation 
      และ Coefficient of Variation ช่วยวัดความผันผวนของราคา
    - **ตัดสินใจบนข้อมูล** — ลดการตัดสินใจตามอารมณ์ 
      โดยใช้หลักฐานเชิงสถิติ (Evidence-Based Decision)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
    <h4>💼 ด้านธุรกิจ (Business)</h4>
    
    - **ประยุกต์กับสินทรัพย์อื่น** — Framework เดียวกันใช้ได้กับ 
      หุ้น, คริปโตเคอร์เรนซี, สินค้าโภคภัณฑ์ (น้ำมัน, เงิน)
    - **Reusable Pipeline** — โค้ดออกแบบเป็น Module 
      สามารถเปลี่ยน Dataset แล้วรันวิเคราะห์ได้ทันที
    - **Dashboard พร้อมใช้** — Streamlit App แสดงผล Interactive 
      สามารถใช้นำเสนอผู้บริหารหรือลูกค้าได้โดยตรง
    - **ลดต้นทุนการวิเคราะห์** — ระบบอัตโนมัติลดเวลาและแรงงาน 
      ในการวิเคราะห์ซ้ำกับข้อมูลชุดใหม่
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-box">
    <h4>🎓 ด้านวิชาการ (Academic)</h4>
    
    - **เรียนรู้ Time Series Analysis** — ได้ฝึกใช้หลายวิธีกับข้อมูลจริง (Real-World Data) 
      ไม่ใช่แค่ตัวอย่างในตำรา
    - **เปรียบเทียบวิธีทางสถิติ** — เข้าใจข้อดี-ข้อเสียของแต่ละวิธีในทางปฏิบัติ:
        - SMA: เรียบง่ายแต่ล่าช้า (Lagging)
        - Holt/Holt-Winters: จับ Trend/Seasonal ได้ดี
        - ARIMA: ยืดหยุ่นแต่ต้อง Tune Parameter
        - Prophet: ใช้งานง่าย เหมาะกับ Seasonal Data
    - **Model Evaluation** — เรียนรู้การวัดผลด้วย MAE, RMSE, MAPE 
      ซึ่งเป็นมาตรฐานในงานพยากรณ์
    - **Data Visualization** — ได้ฝึกสร้างกราฟที่สื่อสารข้อมูลได้ชัดเจน 
      ด้วย Plotly, Apache ECharts, Altair
    - **End-to-End Project** — ฝึกทำโปรเจคตั้งแต่ Clean Data → Analyze → Visualize → Present
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box">
    <h4>🔧 ด้านเทคนิค (Technical Skills)</h4>
    
    | เครื่องมือ | ทักษะที่ได้ |
    |-----------|-----------|
    | **Python** | Data manipulation, Statistical modeling |
    | **Pandas** | DataFrame operations, Resampling |
    | **Statsmodels** | ARIMA, Exponential Smoothing |
    | **Prophet** | Automated time series forecasting |
    | **Plotly / ECharts** | Interactive visualization |
    | **Streamlit** | Web application development |
    | **Jamovi** | Statistical analysis (GUI-based) |
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────
# ECHARTS: Gauge showing project completion
# ──────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 สรุปขอบเขตของโครงงาน")

gauge_opt = {
    "series": [{
        "type": "gauge",
        "startAngle": 180,
        "endAngle": 0,
        "min": 0,
        "max": 7,
        "splitNumber": 7,
        "axisLine": {
            "lineStyle": {
                "width": 20,
                "color": [
                    [0.14, "#FF6B6B"],
                    [0.29, "#FFE66D"],
                    [0.43, "#4ECDC4"],
                    [0.57, "#45B7D1"],
                    [0.71, "#96CEB4"],
                    [0.86, "#FFEAA7"],
                    [1.0, "#FFD700"],
                ],
            }
        },
        "pointer": {"itemStyle": {"color": "#FFD700"}},
        "axisTick": {"distance": -20, "length": 8, "lineStyle": {"color": "#999"}},
        "splitLine": {"distance": -25, "length": 15, "lineStyle": {"color": "#999"}},
        "axisLabel": {"color": "#CCC", "distance": -35, "fontSize": 11},
        "detail": {"valueAnimation": True, "formatter": "{value} / 7 หัวข้อ",
                   "color": "#FFD700", "fontSize": 16, "offsetCenter": [0, "60%"]},
        "data": [{"value": 7, "name": "ครบทุกหัวข้อตามโจทย์"}],
        "title": {"offsetCenter": [0, "85%"], "fontSize": 13, "color": "#AAA"},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=gauge_opt, height="350px")

st.markdown("---")
st.caption("หน้า 3/7 — ประโยชน์จากผลการวิเคราะห์")
