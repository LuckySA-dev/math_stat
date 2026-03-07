"""
Page 2: วัตถุประสงค์
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from streamlit_echarts import st_echarts
from helpers import inject_css

inject_css()

st.markdown('<div class="page-title">🎯 2. วัตถุประสงค์</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="section-box">
<h3>เป้าหมายของโครงงาน</h3>

โครงงานนี้มีวัตถุประสงค์เพื่อศึกษาและประยุกต์ใช้เทคนิคทางสถิติ 
ในการวิเคราะห์ข้อมูลอนุกรมเวลา (Time Series) ของราคาทองคำ XAU/USD 
โดยมีรายละเอียดดังนี้:
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────
# วัตถุประสงค์แต่ละข้อ พร้อมคำอธิบาย
# ──────────────────────────────────────
objectives = [
    {
        "icon": "📊",
        "title": "ศึกษาลักษณะข้อมูลด้วยสถิติเชิงพรรณนา",
        "desc": """
        วิเคราะห์ข้อมูลราคาทองคำ XAU/USD ย้อนหลัง 10 ปี (2016–2025) ด้วยสถิติเบื้องต้น ได้แก่
        ค่าเฉลี่ย (Mean), ส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation), มัธยฐาน (Median), 
        ค่าต่ำสุด-สูงสุด (Min-Max), ความเบ้ (Skewness), ความโด่ง (Kurtosis) 
        เพื่อทำความเข้าใจลักษณะการกระจายตัวของข้อมูลก่อนทำการพยากรณ์
        """,
    },
    {
        "icon": "🔍",
        "title": "วิเคราะห์องค์ประกอบของอนุกรมเวลา (Time Series Decomposition)",
        "desc": """
        แยกข้อมูลรายเดือนออกเป็น 3 องค์ประกอบ:
        - **Trend** — แนวโน้มระยะยาวของราคา (ขาขึ้นหรือขาลง)
        - **Seasonal** — รูปแบบที่เกิดซ้ำเป็นวัฏจักร (ถ้ามี)
        - **Residual** — ความผันผวนที่เหลือซึ่งไม่สามารถอธิบายด้วย Trend และ Seasonal
        """,
    },
    {
        "icon": "⚖️",
        "title": "เปรียบเทียบวิธีพยากรณ์อย่างน้อย 4 วิธี",
        "desc": """
        ทดสอบและเปรียบเทียบประสิทธิภาพของ **5 วิธีพยากรณ์**:
        
        | # | วิธี | ประเภท | คำอธิบาย |
        |---|------|--------|---------|
        | 1 | **SMA** (Simple Moving Average) | Moving Average | ค่าเฉลี่ยเคลื่อนที่แบบง่าย ใช้ข้อมูลย้อนหลัง 3 เดือน |
        | 2 | **Holt** (Double Exp. Smoothing) | Exponential Smoothing | ปรับเรียบแบบ 2 ชั้น (Level + Trend) |
        | 3 | **Holt-Winters** (Triple Exp. Smoothing) | Exponential Smoothing | ปรับเรียบแบบ 3 ชั้น (Level + Trend + Seasonal) |
        | 4 | **ARIMA** (AutoRegressive Integrated Moving Average) | Box-Jenkins | รวม AR, I, MA เข้าด้วยกัน |
        | 5 | **Prophet** (Facebook/Meta) | Additive Model | แยก Trend + Seasonal อัตโนมัติ |
        """,
    },
    {
        "icon": "🏆",
        "title": "เลือกแบบจำลองที่ดีที่สุด",
        "desc": """
        เปรียบเทียบทุกวิธีด้วย **3 ตัวชี้วัด** (Evaluation Metrics):
        - **MAE** (Mean Absolute Error) — ค่าเฉลี่ยความผิดพลาดสัมบูรณ์ (หน่วย: USD)
        - **RMSE** (Root Mean Squared Error) — รากที่สองของค่าเฉลี่ยกำลังสองของความผิดพลาด
        - **MAPE** (Mean Absolute Percentage Error) — ค่าเปอร์เซ็นต์ความผิดพลาดเฉลี่ย
        
        แบบจำลองที่ให้ค่า MAPE ต่ำที่สุดจะถูกเลือกเป็น **Best Model**
        """,
    },
    {
        "icon": "📈",
        "title": "สร้างภาพประกอบเปรียบเทียบข้อมูลจริง vs ข้อมูลทำนาย",
        "desc": """
        แสดงผลด้วยกราฟเส้น (Line Chart) ที่ซ้อนราคาจริง (Actual) กับราคาที่พยากรณ์ได้ (Predicted)
        จากทุกวิธี เพื่อให้เห็นภาพความแม่นยำของแต่ละแบบจำลองอย่างชัดเจน
        """,
    },
]

for i, obj in enumerate(objectives):
    st.markdown(f"""
    <div class="section-box">
        <h4>{obj['icon']} วัตถุประสงค์ข้อ {i+1}: {obj['title']}</h4>
        {obj['desc']}
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────
# ECHARTS: workflow diagram (tree)
# ──────────────────────────────────────
st.markdown("---")
st.markdown("### 🔄 แผนผังขั้นตอนการทำงาน")

tree_opt = {
    "tooltip": {"trigger": "item", "triggerOn": "mousemove"},
    "series": [{
        "type": "tree",
        "data": [{
            "name": "XAU/USD Analysis",
            "children": [
                {"name": "1. เก็บข้อมูล\n(Yahoo Finance)", "children": [
                    {"name": "Clean Data"},
                    {"name": "2,512 Records"},
                ]},
                {"name": "2. สถิติเบื้องต้น", "children": [
                    {"name": "Descriptive Stats"},
                    {"name": "Distribution"},
                ]},
                {"name": "3. Decomposition", "children": [
                    {"name": "Trend"},
                    {"name": "Seasonal"},
                    {"name": "Residual"},
                ]},
                {"name": "4. เปรียบเทียบ 5 วิธี", "children": [
                    {"name": "SMA"},
                    {"name": "Holt"},
                    {"name": "Holt-Winters"},
                    {"name": "ARIMA"},
                    {"name": "Prophet"},
                ]},
                {"name": "5. เลือก Best Model", "children": [
                    {"name": "MAE / RMSE / MAPE"},
                ]},
            ]
        }],
        "left": "2%", "right": "2%", "top": "10%", "bottom": "10%",
        "orient": "vertical",
        "expandAndCollapse": True,
        "initialTreeDepth": 2,
        "label": {"position": "top", "verticalAlign": "middle", "fontSize": 12},
        "leaves": {"label": {"position": "bottom"}},
        "animationDuration": 550,
        "animationDurationUpdate": 750,
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=tree_opt, height="500px")

st.markdown("---")
st.caption("หน้า 2/7 — วัตถุประสงค์")
