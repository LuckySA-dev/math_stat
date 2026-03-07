"""
Page 4: สถิติเบื้องต้นบรรยายข้อมูล ที่มาของข้อมูลพร้อมอ้างอิง
— ฉบับเจาะลึก: แสดงสูตร หลักการ และการตีความของแต่ละค่าสถิติ
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
from scipy.stats import jarque_bera, gaussian_kde
from helpers import inject_css, get_data, descriptive_stats

inject_css()
df, monthly = get_data()
s = df["price"]          # Close Price series
n = len(s)
mean_val = s.mean()
med_val  = s.median()
std_val  = s.std()       # sample std (ddof=1)
var_val  = s.var()       # sample variance (ddof=1)
skew_val = s.skew()
kurt_val = s.kurtosis()
q1       = s.quantile(0.25)
q3       = s.quantile(0.75)
iqr_val  = q3 - q1
range_val = s.max() - s.min()
cv_val   = (std_val / mean_val) * 100

st.markdown(
    '<div class="page-title">📊 4. สถิติเบื้องต้นบรรยายข้อมูล</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ═══════════════════════════════════════════════
#  ที่มาข้อมูล
# ═══════════════════════════════════════════════
st.markdown("""
<div class="section-box">
<h3>📂 ที่มาของข้อมูลและอ้างอิง</h3>

| รายการ | รายละเอียด |
|--------|-----------|
| **ชื่อชุดข้อมูล** | XAU/USD Historical Data (Gold Spot Price in USD) |
| **แหล่งที่มา** | [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF/) (GC=F Gold Futures) |
| **ช่วงเวลา** | 4 มกราคม 2016 – 30 ธันวาคม 2025 (10 ปี) |
| **ความถี่** | รายวัน (Daily) — เฉพาะวันทำการ |
| **จำนวนข้อมูล** | **2,512 แถว** × 5 ตัวแปร |
| **ไฟล์** | `datasets/xauusd_2016-2025.csv` |

**ตัวแปรในชุดข้อมูล:**
- `date` — วันที่ (MM/DD/YYYY)
- `price` — ราคาปิด (Close Price) หน่วย USD ต่อ Troy Ounce
- `open` — ราคาเปิด
- `high` — ราคาสูงสุดของวัน
- `low` — ราคาต่ำสุดของวัน
- `change %` — เปอร์เซ็นต์การเปลี่ยนแปลงจากวันก่อนหน้า
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  A. ค่าแนวโน้มเข้าสู่ส่วนกลาง
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📐 A. ค่าแนวโน้มเข้าสู่ส่วนกลาง (Measures of Central Tendency)")
st.markdown("""
<div class="section-box">
กลุ่มค่าสถิติที่ใช้เป็น <b>ตัวแทน</b> ของข้อมูลทั้งชุด —
บอกว่าข้อมูลส่วนใหญ่ "อยู่บริเวณไหน"
</div>
""", unsafe_allow_html=True)

# ── 1. Mean ──────────────────────────
st.markdown("### 1. ค่าเฉลี่ยเลขคณิต (Arithmetic Mean)")
st.latex(
    r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i "
    r"= \frac{x_1 + x_2 + \cdots + x_n}{n}"
)
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> ค่าเฉลี่ยเลขคณิต คือ ผลรวมของข้อมูลทั้งหมดหารด้วยจำนวนข้อมูล
เป็นค่าที่ใช้บ่อยที่สุดในการหาตำแหน่งจุดศูนย์กลางของข้อมูล
แต่มีข้อจำกัดคือ <b>ไวต่อค่าสุดโต่ง (Outlier)</b>
</div>
""", unsafe_allow_html=True)
st.latex(
    rf"\bar{{x}} = \frac{{\sum x_i}}{{{n}}} "
    rf"= \frac{{{s.sum():,.2f}}}{{{n}}} "
    rf"= \mathbf{{{mean_val:,.2f}}} \;\text{{ USD}}"
)
st.info(
    f"**ตีความ:** ราคาปิดเฉลี่ยตลอด 10 ปี = **${mean_val:,.2f}** "
    "→ ใช้เป็นจุดอ้างอิงกลางสำหรับเปรียบเทียบกับราคาปัจจุบัน"
)

# ── 2. Median ────────────────────────
st.markdown("### 2. มัธยฐาน (Median)")
st.latex(
    r"\widetilde{x} = \begin{cases}"
    r" x_{\left(\frac{n+1}{2}\right)} & \text{ถ้า } n \text{ เป็นจำนวนคี่} \\"
    r" \dfrac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2}"
    r" & \text{ถ้า } n \text{ เป็นจำนวนคู่}"
    r"\end{cases}"
)
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> มัธยฐาน คือค่ากลางเมื่อเรียงข้อมูลจากน้อยไปมาก —
<b>ไม่ถูกกระทบจาก Outlier</b> ต่างจาก Mean<br>
ข้อมูลนี้ <i>n</i> = {n} (คู่) → หาค่าตำแหน่งที่ {n//2} กับ {n//2+1} แล้วเฉลี่ย
</div>
""", unsafe_allow_html=True)
st.latex(rf"\widetilde{{x}} = \mathbf{{{med_val:,.2f}}} \;\text{{ USD}}")

diff_pct = ((mean_val - med_val) / med_val) * 100
st.info(
    f"**ตีความ:** Mean ({mean_val:,.2f}) **{'>' if mean_val > med_val else '<'}** "
    f"Median ({med_val:,.2f}) ห่างกัน {abs(diff_pct):.1f}% "
    f"→ ข้อมูล **{'เบ้ขวา (Right-Skewed) มีหางค่าสูงยาว' if diff_pct > 0 else 'เบ้ซ้าย'}**"
)

# ── 3. Mode ──────────────────────────
mode_val = s.mode().iloc[0] if not s.mode().empty else None
st.markdown("### 3. ฐานนิยม (Mode)")
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> ฐานนิยม คือค่าที่ปรากฏบ่อยที่สุดในชุดข้อมูล
สำหรับข้อมูลต่อเนื่อง (ราคาหุ้น/ทองคำ) มักใช้ <b>Mode ของข้อมูลที่จัดกลุ่มแล้ว (Binned)</b> แทน<br>
Mode = <b>${mode_val:,.2f}</b>
</div>
""", unsafe_allow_html=True)

# ── Summary Cards ────────────────────
st.markdown("#### 📊 สรุป Central Tendency")
cc = st.columns(3)
cc[0].metric("Mean (ค่าเฉลี่ย)", f"${mean_val:,.2f}")
cc[1].metric("Median (มัธยฐาน)", f"${med_val:,.2f}")
cc[2].metric("Mode (ฐานนิยม)", f"${mode_val:,.2f}" if mode_val else "N/A")

# ═══════════════════════════════════════════════
#  B. ค่าการกระจาย (Dispersion)
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📏 B. ค่าการกระจายของข้อมูล (Measures of Dispersion)")
st.markdown("""
<div class="section-box">
ค่าการกระจาย บอกว่าข้อมูล <b>กระจายตัวออกจากค่ากลาง</b> มากน้อยแค่ไหน —
ใช้วัด <b>ความเสี่ยง (Volatility)</b> ของราคาทองคำ
</div>
""", unsafe_allow_html=True)

# ── 4. Range ─────────────────────────
st.markdown("### 4. พิสัย (Range)")
st.latex(r"\text{Range} = x_{\max} - x_{\min}")
st.latex(
    rf"\text{{Range}} = {s.max():,.2f} - {s.min():,.2f} "
    rf"= \mathbf{{{range_val:,.2f}}} \;\text{{ USD}}"
)
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> Range = ค่ามากสุด − ค่าน้อยสุด
— บอก <b>ช่วงกว้าง</b> ของข้อมูล แต่ไวต่อ Outlier<br>
<b>ตีความ:</b> ราคาทองคำเคลื่อนที่ในช่วงกว้าง <b>${range_val:,.2f}</b> ตลอด 10 ปี
</div>
""", unsafe_allow_html=True)

# ── 5. Variance ──────────────────────
st.markdown("### 5. ความแปรปรวน (Variance)")
st.latex(
    r"s^{2} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^{2}"
)
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> Variance = ค่าเฉลี่ยของ <i>กำลังสอง</i> ของความเบี่ยงเบนจาก Mean<br>
ใช้ $n-1$ (Bessel's Correction) เพื่อให้เป็น <b>Unbiased Estimator</b> สำหรับ Sample<br><br>
$s^2 = {var_val:,.2f}$ &nbsp; (หน่วย: USD²)
</div>
""", unsafe_allow_html=True)

# ── 6. Std Dev ───────────────────────
st.markdown("### 6. ส่วนเบี่ยงเบนมาตรฐาน (Standard Deviation)")
st.latex(
    r"s = \sqrt{s^{2}} "
    r"= \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^{2}}"
)
st.latex(
    rf"s = \sqrt{{{var_val:,.2f}}} = \mathbf{{{std_val:,.2f}}} \;\text{{ USD}}"
)
st.markdown(f"""
<div class="highlight-box">
<b>ตีความ:</b> ราคาทองคำเบี่ยงเบนจาก Mean โดยเฉลี่ย <b>±${std_val:,.2f}</b><br>
ถ้าข้อมูลเป็น Normal → ~68% อยู่ใน
<b>${mean_val - std_val:,.2f} – ${mean_val + std_val:,.2f}</b>
</div>
""", unsafe_allow_html=True)

# ── 7. IQR ───────────────────────────
st.markdown("### 7. พิสัยระหว่างควอร์ไทล์ (IQR)")
st.latex(r"\text{IQR} = Q_3 - Q_1")
st.latex(
    rf"\text{{IQR}} = {q3:,.2f} - {q1:,.2f} "
    rf"= \mathbf{{{iqr_val:,.2f}}} \;\text{{ USD}}"
)
lower_fence = q1 - 1.5 * iqr_val
upper_fence = q3 + 1.5 * iqr_val
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> IQR ครอบคลุมข้อมูล <b>50% ตรงกลาง</b> (Q1 ถึง Q3)
— <b>Robust</b> ต่อค่าผิดปกติ<br><br>
<b>Tukey's Fence (ใช้หา Outlier):</b><br>
• Lower Fence = Q1 − 1.5×IQR = {q1:,.2f} − {1.5*iqr_val:,.2f} = <b>{lower_fence:,.2f}</b><br>
• Upper Fence = Q3 + 1.5×IQR = {q3:,.2f} + {1.5*iqr_val:,.2f} = <b>{upper_fence:,.2f}</b><br>
ข้อมูลนอก Fence → Outlier
</div>
""", unsafe_allow_html=True)

# ── 8. CV ────────────────────────────
st.markdown("### 8. สัมประสิทธิ์การแปรผัน (CV)")
st.latex(
    r"\text{CV} = \frac{s}{\bar{x}} \times 100\%"
)
st.latex(
    rf"\text{{CV}} = \frac{{{std_val:,.2f}}}{{{mean_val:,.2f}}} "
    rf"\times 100\% = \mathbf{{{cv_val:.2f}\%}}"
)
st.markdown(f"""
<div class="highlight-box">
<b>หลักการ:</b> CV = อัตราส่วน Std ต่อ Mean (เป็น %)
— ใช้เปรียบเทียบความกระจายระหว่างชุดข้อมูลที่มีหน่วยต่างกัน<br>
<b>ตีความ:</b> CV = {cv_val:.2f}% → ผันผวนค่อนข้างสูง
(เนื่องจากมี Strong Uptrend ตลอด 10 ปี)
</div>
""", unsafe_allow_html=True)

# ── Dispersion Cards ─────────────────
st.markdown("#### 📊 สรุปค่าการกระจาย")
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Range", f"${range_val:,.2f}")
d2.metric("Std Dev", f"${std_val:,.2f}")
d3.metric("Variance", f"{var_val:,.0f}")
d4.metric("IQR", f"${iqr_val:,.2f}")
d5.metric("CV", f"{cv_val:.2f}%")

# ═══════════════════════════════════════════════
#  C. รูปร่างการแจกแจง (Shape)
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 C. รูปร่างของการแจกแจง (Measures of Shape)")

# ── 9. Skewness ──────────────────────
st.markdown("### 9. ความเบ้ (Skewness)")
st.latex(
    r"\text{Skewness} = \frac{n}{(n-1)(n-2)}"
    r"\sum_{i=1}^{n}\!\left(\frac{x_i - \bar{x}}{s}\right)^{\!3}"
)
st.latex(rf"\text{{Skewness}} = \mathbf{{{skew_val:.4f}}}")
st.markdown(f"""
<div class="section-box">
<b>หลักการ:</b> ความเบ้วัดว่าการแจกแจง <b>เอียง</b> ไปทางไหน

| ค่า | ความหมาย |
|:---:|:---|
| = 0 | สมมาตร (Symmetric) |
| > 0 | **เบ้ขวา** — หางขวายาว |
| < 0 | เบ้ซ้าย — หางซ้ายยาว |

<b>ข้อมูลนี้:</b> Skewness = <b>{skew_val:.4f}</b> > 0
→ <b style="color:#FFD700;">เบ้ขวา</b> — ราคาส่วนใหญ่กระจุกที่ระดับต่ำ–กลาง
แต่ช่วง 2024–2025 ดันหางขวาออกไปไกล
</div>
""", unsafe_allow_html=True)

# ── 10. Kurtosis ─────────────────────
st.markdown("### 10. ความโด่ง (Kurtosis)")
st.latex(
    r"\text{Excess Kurtosis} = "
    r"\frac{n(n+1)}{(n-1)(n-2)(n-3)}"
    r"\sum\!\left(\frac{x_i-\bar{x}}{s}\right)^{\!4}"
    r"- \frac{3(n-1)^{2}}{(n-2)(n-3)}"
)
st.latex(rf"\text{{Excess Kurtosis}} = \mathbf{{{kurt_val:.4f}}}")
st.markdown(f"""
<div class="section-box">
<b>หลักการ:</b> Kurtosis วัดความ "หนา" ของหาง
(Excess Kurtosis ลบ 3 ของ Normal ออกแล้ว)

| ค่า | ชื่อ | ความหมาย |
|:---:|:---:|:---|
| = 0 | Mesokurtic | เหมือน Normal |
| > 0 | **Leptokurtic** | หางหนา → Extreme Values มากกว่า |
| < 0 | Platykurtic | หางเบา → ข้อมูลกระจุกใกล้กลาง |

<b>ข้อมูลนี้:</b> = <b>{kurt_val:.4f}</b> {'> 0' if kurt_val > 0 else '< 0'}
→ <b style="color:#FFD700;">{'Leptokurtic' if kurt_val > 0 else 'Platykurtic'}</b>
{'→ มีโอกาสเกิด Extreme Price Movement สูงกว่า Normal' if kurt_val > 0 else ''}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  D. Jarque-Bera Test
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔬 D. ทดสอบการแจกแจงปกติ (Jarque-Bera Test)")

jb_stat, jb_p = jarque_bera(s.dropna())
st.latex(
    r"\text{JB} = \frac{n}{6}"
    r"\!\left( S^{2} + \frac{(K-3)^{2}}{4} \right)"
)
st.markdown("โดย $S$ = Skewness, $K$ = Kurtosis ปกติ (Fisher + 3)")
fisher_k = kurt_val + 3
st.latex(
    rf"\text{{JB}} = \frac{{{n}}}{{6}}"
    rf"\left({skew_val:.4f}^2 + \frac{{({fisher_k:.4f}-3)^2}}{{4}}\right)"
    rf"= \mathbf{{{jb_stat:,.2f}}}"
)
reject = jb_p < 0.05
st.markdown(f"""
<div class="section-box">
<b>สมมติฐาน:</b><br>
• H₀ : ข้อมูลมีการแจกแจงปกติ (Normal)<br>
• H₁ : ข้อมูลไม่มีการแจกแจงปกติ

| ค่า | ผลลัพธ์ |
|-----|---------|
| **JB Statistic** | {jb_stat:,.2f} |
| **p-value** | {jb_p:.2e} |
| **α** | 0.05 |
| **ผลตัดสิน** | {"**ปฏิเสธ H₀** (p < 0.05)" if reject else "ไม่ปฏิเสธ H₀"} |

{"❌ <b>สรุป:</b> ข้อมูลราคาทองคำ <u>ไม่เป็น Normal Distribution</u>" if reject
 else "✅ ไม่ปฏิเสธว่าเป็น Normal"}
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  E. Descriptive Stats Table
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📋 E. ตารางสถิติเชิงพรรณนาสรุป")
st.markdown("""
<div class="highlight-box">
ตารางรวมค่าสถิติทั้งหมดของตัวแปรราคา 4 ตัว (Price, Open, High, Low)
</div>
""", unsafe_allow_html=True)
stats = descriptive_stats(df)
st.dataframe(
    stats.style.format("{:,.2f}").background_gradient(cmap="YlOrRd", axis=1),
    height=550, use_container_width=True,
)

# ═══════════════════════════════════════════════
#  F. PLOTS — ปรับอ่านง่าย
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 F. กราฟแสดงการกระจายตัวของข้อมูล")

# ── F1: Histogram + KDE (Plotly, full-width) ─
st.markdown("### F1. Histogram + KDE — Close Price")
st.markdown("""
<div class="highlight-box">
ฮิสโตแกรม + เส้น KDE ของราคาปิดรายวัน<br>
เส้นประ <b style="color:#E57373;">แดง</b> = Mean &nbsp;
เส้นประ <b style="color:#4ECDC4;">เขียว</b> = Median
</div>
""", unsafe_allow_html=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=s, nbinsx=50, name="Frequency",
    marker_color="rgba(255,215,0,0.6)",
    marker_line=dict(color="#FFD700", width=0.5),
))
kde = gaussian_kde(s.dropna())
x_range = np.linspace(s.min(), s.max(), 300)
kde_y = kde(x_range) * n * (s.max() - s.min()) / 50
fig_hist.add_trace(go.Scatter(
    x=x_range, y=kde_y, mode="lines", name="KDE",
    line=dict(color="#FFEAA7", width=2.5),
))
fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="#E57373",
                   annotation_text=f"Mean ${mean_val:,.0f}",
                   annotation_font_color="#E57373")
fig_hist.add_vline(x=med_val, line_dash="dash", line_color="#4ECDC4",
                   annotation_text=f"Median ${med_val:,.0f}",
                   annotation_font_color="#4ECDC4")
fig_hist.update_layout(
    template="plotly_dark", height=430,
    title="Close Price Distribution (Histogram + KDE)",
    xaxis_title="Close Price (USD)", yaxis_title="Frequency",
    showlegend=True,
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    bargap=0.03,
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── F2: Box Plot — แนวนอน, full-width, ไม่ notch ──
st.markdown("### F2. Box Plot — OHLC (แนวนอน)")
st.markdown("""
<div class="highlight-box">
Box Plot แนวนอนแสดง Q1 → Median → Q3, Whisker (±1.5×IQR) และจุด Outlier<br>
เพชร (♦) = Mean ± SD &nbsp;|&nbsp; จุดเล็ก = Outlier
</div>
""", unsafe_allow_html=True)

fig_box = go.Figure()
box_items = [
    ("Low",   df["low"],   "#E57373"),
    ("High",  df["high"],  "#81C784"),
    ("Open",  df["open"],  "#4FC3F7"),
    ("Close", df["price"], "#FFD700"),
]
for name, data, color in box_items:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig_box.add_trace(go.Box(
        x=data, name=name, orientation="h",
        marker=dict(color=color, outliercolor=color, size=3, opacity=0.5),
        line=dict(color=color, width=2),
        fillcolor=f"rgba({r},{g},{b},0.18)",
        boxmean="sd",
        boxpoints="outliers",
    ))

fig_box.update_layout(
    template="plotly_dark", height=340,
    title="Box Plot: OHLC (Horizontal · Mean±SD)",
    xaxis_title="Price (USD)",
    yaxis=dict(autorange="reversed"),
    margin=dict(l=90, r=20, t=50, b=40),
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown(f"""
<div class="section-box">
<b>อ่าน Box Plot:</b>

| ส่วนประกอบ | ความหมาย | Close Price |
|:---:|:---|:---|
| กล่อง (Box) | Q1 → Q3 (IQR) | ${q1:,.2f} – ${q3:,.2f} |
| เส้นกลาง | Median | ${med_val:,.2f} |
| เพชร | Mean ± SD | ${mean_val:,.2f} ± ${std_val:,.2f} |
| หนวด (Whisker) | ±1.5×IQR จากกล่อง | {lower_fence:,.2f} – {upper_fence:,.2f} |
| จุดนอก | Outlier | > Whisker |
</div>
""", unsafe_allow_html=True)

# ── F3: Violin Plot — สองแถว 5+5 ปี ──
st.markdown("---")
st.markdown("### F3. Violin Plot — การกระจายตัวรายปี")
st.markdown("""
<div class="highlight-box">
Violin = Box Plot + KDE หมุน 90°<br>
ส่วนกว้าง = ราคาที่พบบ่อย &nbsp;|&nbsp; ส่วนแคบ = พบน้อย<br>
แบ่ง 2 แถว (แถวละ 5 ปี) เพื่อให้ Violin แต่ละปีใหญ่เพียงพอที่จะอ่านรูปร่างได้ชัดเจน
</div>
""", unsafe_allow_html=True)

df_vio = df.copy()
df_vio["year"] = df_vio.index.year
all_years = sorted(df_vio["year"].unique())
year_colors = {
    2016: "#E57373", 2017: "#FFB74D", 2018: "#FFD54F",
    2019: "#81C784", 2020: "#4FC3F7", 2021: "#7986CB",
    2022: "#BA68C8", 2023: "#F06292", 2024: "#FFD700",
    2025: "#4DB6AC",
}

# split into two rows: first 5 years, last 5 years
half = len(all_years) // 2
year_groups = [all_years[:half], all_years[half:]]
group_labels = ["2016 – 2020", "2021 – 2025"]

for gi, (years_g, label_g) in enumerate(zip(year_groups, group_labels)):
    fig_v = go.Figure()
    for yr in years_g:
        yr_data = df_vio[df_vio["year"] == yr]["price"]
        c = year_colors.get(yr, "#AAA")
        r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
        fig_v.add_trace(go.Violin(
            y=yr_data, name=str(yr),
            box_visible=True, meanline_visible=True,
            line_color=c,
            fillcolor=f"rgba({r},{g},{b},0.35)",
            points="outliers", pointpos=0,
            scalemode="width", width=0.8,
        ))
    fig_v.update_layout(
        template="plotly_dark",
        height=420,
        title=f"Violin Plot: Close Price — {label_g}",
        yaxis_title="Close Price (USD)",
        xaxis_title="ปี",
        showlegend=False,
        violinmode="group",
    )
    st.plotly_chart(fig_v, use_container_width=True)

# ── F4: Correlation Heatmap (ECharts) ─
st.markdown("---")
st.markdown("### F4. Correlation Matrix (Heatmap)")
st.markdown("""
<div class="highlight-box">
<b>Pearson Correlation</b> วัดความสัมพันธ์เชิงเส้นระหว่าง 2 ตัวแปร (−1 ถึง +1)
</div>
""", unsafe_allow_html=True)
st.latex(
    r"r_{xy} = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}"
    r"{\sqrt{\sum(x_i-\bar{x})^{2}\;\sum(y_i-\bar{y})^{2}}}"
)

cols = ["price", "open", "high", "low"]
corr = df[cols].corr()
corr_data = []
for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols):
        corr_data.append([i, j, round(corr.loc[c1, c2], 4)])

heatmap_opt = {
    "tooltip": {"position": "top", "formatter": "{c}"},
    "xAxis": {"type": "category",
              "data": [c.capitalize() for c in cols],
              "splitArea": {"show": True}},
    "yAxis": {"type": "category",
              "data": [c.capitalize() for c in cols],
              "splitArea": {"show": True}},
    "visualMap": {
        "min": 0.998, "max": 1.0, "calculable": True,
        "orient": "horizontal", "left": "center", "bottom": "0%",
        "inRange": {"color": ["#1a1a2e", "#FFD700", "#FF4500"]},
    },
    "series": [{
        "type": "heatmap", "data": corr_data,
        "label": {"show": True, "fontSize": 14, "formatter": "{@[2]}"},
        "emphasis": {"itemStyle": {"shadowBlur": 10,
                                   "shadowColor": "rgba(0,0,0,0.5)"}},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=heatmap_opt, height="400px")

st.markdown("""
<div class="section-box">
<b>ตีความ:</b> Correlation ระหว่าง OHLC สูงมาก (>0.999)
เนื่องจาก Open/High/Low มักเคลื่อนไหวตาม Close —
ลักษณะปกติของข้อมูลราคาสินทรัพย์
</div>
""", unsafe_allow_html=True)

# ── F5: Yearly Stats Table ───────────
st.markdown("---")
st.markdown("### F5. สถิติรายปี")
st.markdown("""
<div class="highlight-box">
ตารางสรุป Mean, Min, Max, Std ของแต่ละปี
</div>
""", unsafe_allow_html=True)

yearly = df.groupby(df.index.year)["price"].agg(
    ["mean", "min", "max", "std", "count"]
)
yearly.columns = ["Mean ($)", "Min ($)", "Max ($)", "Std ($)", "วันทำการ"]
yearly.index.name = "ปี"
st.dataframe(
    yearly.style.format({
        "Mean ($)": "{:,.2f}", "Min ($)": "{:,.2f}",
        "Max ($)": "{:,.2f}", "Std ($)": "{:,.2f}",
        "วันทำการ": "{:,.0f}",
    }).background_gradient(cmap="YlOrRd", subset=["Mean ($)"]),
    use_container_width=True,
)

# ── F6: Monthly Heatmap (ECharts) ────
st.markdown("---")
st.markdown("### F6. Monthly Average Price Heatmap")
st.markdown("""
<div class="highlight-box">
สีเข้ม = ราคาสูง &nbsp;|&nbsp; เห็นชัดว่าครึ่งหลัง 2024 − 2025 มีสีเข้มที่สุด
</div>
""", unsafe_allow_html=True)

df_temp = df.copy()
df_temp["year"]  = df_temp.index.year
df_temp["month"] = df_temp.index.month
pivot = df_temp.pivot_table(
    values="price", index="year", columns="month", aggfunc="mean"
)
months_lbl = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
years_lbl = [str(y) for y in pivot.index]
hm_data = []
for i, year in enumerate(pivot.index):
    for j in range(1, 13):
        val = pivot.loc[year, j] if j in pivot.columns else None
        if val is not None and not np.isnan(val):
            hm_data.append([j - 1, i, round(val, 0)])

monthly_hm_opt = {
    "tooltip": {"position": "top",
                "formatter": "ปี {b1} เดือน {b0}: ${c} USD"},
    "xAxis": {"type": "category", "data": months_lbl,
              "splitArea": {"show": True}},
    "yAxis": {"type": "category", "data": years_lbl,
              "splitArea": {"show": True}},
    "visualMap": {
        "min": 900, "max": 4500, "calculable": True,
        "orient": "horizontal", "left": "center", "bottom": "0%",
        "inRange": {"color": ["#1a1a2e", "#2d1b69", "#8b5cf6",
                              "#FFD700", "#FF4500"]},
    },
    "series": [{
        "type": "heatmap", "data": hm_data,
        "label": {"show": True, "fontSize": 10, "formatter": "{@[2]}"},
        "emphasis": {"itemStyle": {"shadowBlur": 10}},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=monthly_hm_opt, height="380px")

# ── F7: Daily Returns Distribution (Plotly) ──
st.markdown("---")
st.markdown("### F7. Daily Returns Distribution")
st.markdown("""
<div class="highlight-box">
Histogram + KDE ของ Daily Change (%)
— เกือบ Normal แต่มี Fat Tails (Leptokurtic)
</div>
""", unsafe_allow_html=True)

returns = (df["change %"] * 100).dropna()
fig_ret = go.Figure()
fig_ret.add_trace(go.Histogram(
    x=returns, nbinsx=80, name="Daily Change",
    marker_color="rgba(78,205,196,0.55)",
    marker_line=dict(color="#4ECDC4", width=0.5),
))
kde_r = gaussian_kde(returns.values)
x_ret = np.linspace(returns.min(), returns.max(), 300)
kde_ry = kde_r(x_ret) * len(returns) * (returns.max() - returns.min()) / 80
fig_ret.add_trace(go.Scatter(
    x=x_ret, y=kde_ry, mode="lines", name="KDE",
    line=dict(color="#FFEAA7", width=2),
))
ret_mean = returns.mean()
fig_ret.add_vline(x=ret_mean, line_dash="dash", line_color="#E57373",
                  annotation_text=f"Mean {ret_mean:.3f}%")
fig_ret.update_layout(
    template="plotly_dark", height=380,
    title="Distribution of Daily Returns (%)",
    xaxis_title="Daily Change (%)", yaxis_title="Frequency",
    showlegend=True,
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    bargap=0.02,
)
st.plotly_chart(fig_ret, use_container_width=True)

ret_skew = returns.skew()
ret_kurt = returns.kurtosis()
jb_r, jb_rp = jarque_bera(returns.values)
st.markdown(f"""
<div class="section-box">
<b>สถิติ Daily Returns:</b>

| ค่า | ผลลัพธ์ | ความหมาย |
|:---|:---|:---|
| Mean | {ret_mean:.4f}% | ผลตอบแทนเฉลี่ยรายวัน |
| Std | {returns.std():.4f}% | ความผันผวนรายวัน |
| Skewness | {ret_skew:.4f} | {"เบ้ซ้ายเล็กน้อย" if ret_skew < 0 else "เบ้ขวาเล็กน้อย"} |
| Kurtosis | {ret_kurt:.4f} | Leptokurtic → Fat Tails |
| JB Test | {jb_r:,.2f} (p = {jb_rp:.2e}) | ไม่เป็น Normal |
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("หน้า 4/7 — สถิติเบื้องต้นบรรยายข้อมูล ที่มาของข้อมูลพร้อมอ้างอิง")
