"""
Page 6: สรุปรูปแบบที่ดีที่สุด
"""
import streamlit as st
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import pandas as pd
from helpers import inject_css, get_models, get_models_compare

inject_css()

st.markdown('<div class="page-title">🏆 6. สรุปรูปแบบที่ดีที่สุด</div>', unsafe_allow_html=True)
st.markdown("---")

test_months = st.slider("🔧 จำนวนเดือนทดสอบ", 3, 12, 6, key="p6_slider")
train, test, results, metrics_df, model_params = get_models(test_months)

best = metrics_df.iloc[0]
best_name = best["method"]

# ──────────────────────────────────────
# WINNER ANNOUNCEMENT
# ──────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(255,215,0,0.15), rgba(255,165,0,0.08));
            border: 2px solid rgba(255,215,0,0.4); border-radius: 16px; padding: 2rem;
            text-align: center; margin: 1rem 0;">
    <h1 style="margin:0; font-size:3rem;">🥇</h1>
    <h2 style="background: linear-gradient(90deg, #FFD700, #FFA500);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               margin: 0.5rem 0;">แบบจำลองที่ดีที่สุด</h2>
    <h1 style="color: #FFD700; margin: 0.5rem 0;">{best_name}</h1>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("MAE (Mean Absolute Error)", f"${best['MAE']:,.2f}",
              help="ค่าเฉลี่ยความผิดพลาดสัมบูรณ์ — หน่วย USD")
with c2:
    st.metric("RMSE (Root Mean Squared Error)", f"${best['RMSE']:,.2f}",
              help="รากที่สองของค่าเฉลี่ยกำลังสองของความผิดพลาด")
with c3:
    st.metric("MAPE (% Error)", f"{best['MAPE (%)']:.2f}%",
              help="เปอร์เซ็นต์ความผิดพลาดเฉลี่ย — ยิ่งน้อยยิ่งดี")

# ──────────────────────────────────────
# INTERPRETATION
# ──────────────────────────────────────
st.markdown(f"""
<div class="section-box">
<h3>📝 การตีความผล</h3>

**{best_name}** ให้ค่า MAPE ต่ำที่สุดที่ **{best['MAPE (%)']:.2f}%** หมายความว่า
โดยเฉลี่ยแล้ว การพยากรณ์ราคาทองคำรายเดือนจะผิดพลาดเพียง **{best['MAPE (%)']:.2f}%** 
จากราคาจริง

**ตัวอย่างเชิงปฏิบัติ:**
- ถ้าราคาจริง = $4,000 → การพยากรณ์จะอยู่ในช่วงประมาณ 
  **${4000 * (1 - best['MAPE (%)'] / 100):,.0f} – ${4000 * (1 + best['MAPE (%)'] / 100):,.0f}**
- MAE = ${best['MAE']:,.2f} หมายความว่าค่าเฉลี่ยของ |ราคาจริง - ราคาทำนาย| = ${best['MAE']:,.2f}

**เหตุผลที่ {best_name} ชนะ:**
- จับแนวโน้ม (Trend) ของราคาทองคำที่ขาขึ้นได้ดี
- ไม่ Overfit หรือ Underfit กับข้อมูล Training
- ให้ค่า MAPE ต่ำที่สุดในทุกตัวชี้วัด
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────
# RANKING TABLE
# ──────────────────────────────────────
st.markdown("### 📊 อันดับทุกวิธี")

medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
for i, (_, row) in enumerate(metrics_df.iterrows()):
    medal = medals[i] if i < len(medals) else f"{i+1}."

    if i == 0:
        bg = "rgba(255,215,0,0.1)"
        border = "rgba(255,215,0,0.3)"
    elif i == 1:
        bg = "rgba(192,192,192,0.08)"
        border = "rgba(192,192,192,0.2)"
    elif i == 2:
        bg = "rgba(205,127,50,0.08)"
        border = "rgba(205,127,50,0.2)"
    else:
        bg = "rgba(100,100,100,0.05)"
        border = "rgba(100,100,100,0.15)"

    st.markdown(f"""
    <div style="background: {bg}; border: 1px solid {border}; border-radius: 10px;
                padding: 0.8rem 1.2rem; margin: 0.4rem 0; display: flex; align-items: center;">
        <span style="font-size: 1.5rem; margin-right: 1rem;">{medal}</span>
        <span style="flex: 1; font-weight: 600; font-size: 1.1rem;">{row['method']}</span>
        <span style="margin: 0 1rem; color: #4ECDC4;">MAE: ${row['MAE']:,.2f}</span>
        <span style="margin: 0 1rem; color: #45B7D1;">RMSE: ${row['RMSE']:,.2f}</span>
        <span style="margin: 0 1rem; color: #FFD700;">MAPE: {row['MAPE (%)']:.2f}%</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────
# RADAR via ECharts
# ──────────────────────────────────────
st.markdown("### 🕸️ Radar Chart: Normalized Error Comparison")
st.markdown("""
<div class="highlight-box">
Radar Chart แสดงค่า Error ที่ Normalize แล้ว (0–1) ของแต่ละวิธี
พื้นที่ยิ่งเล็กยิ่งดี เพราะหมายความว่าค่า Error ทุกตัวต่ำ
</div>
""", unsafe_allow_html=True)

# Normalize
norm = metrics_df.copy()
for col in ["MAE", "RMSE", "MAPE (%)"]:
    mx = norm[col].max()
    norm[col] = (norm[col] / mx * 100).round(1) if mx > 0 else 0

radar_opt = {
    "legend": {"data": norm["method"].tolist(), "orient": "horizontal",
               "bottom": 0, "textStyle": {"color": "#CCC"}},
    "radar": {
        "indicator": [
            {"name": "MAE", "max": 110},
            {"name": "RMSE", "max": 110},
            {"name": "MAPE", "max": 110},
        ],
        "shape": "polygon",
        "splitArea": {"areaStyle": {"color": ["rgba(255,215,0,0.02)", "rgba(255,215,0,0.05)"]}},
        "axisLine": {"lineStyle": {"color": "rgba(255,255,255,0.15)"}},
        "splitLine": {"lineStyle": {"color": "rgba(255,255,255,0.1)"}},
    },
    "series": [{
        "type": "radar",
        "data": [
            {"value": [row["MAE"], row["RMSE"], row["MAPE (%)"]], "name": row["method"]}
            for _, row in norm.iterrows()
        ],
        "areaStyle": {"opacity": 0.15},
    }],
    "backgroundColor": "transparent",
}
st_echarts(options=radar_opt, height="500px")

# ══════════════════════════════════════════════
# COMPARISON: SHORT DATA vs FULL DATA
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 เปรียบเทียบ: ข้อมูลน้อย vs ข้อมูลมาก (Short vs Full Training Data)")
st.markdown("""
<div class="section-box">
ทดสอบว่า <b>ปริมาณข้อมูลที่ใช้ Train</b> ส่งผลต่อความแม่นยำอย่างไร<br>
ทั้งสองกรณีทำนาย <b>ช่วง Test Period เดียวกัน</b> เพื่อให้เปรียบเทียบได้ตรงกัน

| ชุดข้อมูล | รายละเอียด |
|-----------|-----------|
| **Full (10 ปี)** | ใช้ข้อมูลทั้งหมด 2016–2025 เป็น Training Set |
| **Short (5 ปี)** | ใช้เฉพาะข้อมูล 5 ปีล่าสุด (2021–2025) |
</div>
""", unsafe_allow_html=True)

full_result, short_result, n_full_m, n_short_m = get_models_compare(test_months)
_, _, _, metrics_full_c, _ = full_result
_, _, _, metrics_short_c, _ = short_result

col_full, col_short = st.columns(2)
with col_full:
    st.markdown(f"### 📅 Full 10 ปี ({n_full_m} เดือน)")
    best_f = metrics_full_c.iloc[0]
    st.metric("🏆 Best Model", best_f["method"])
    st.metric("Best MAPE", f"{best_f['MAPE (%)']:.2f}%")
    st.dataframe(
        metrics_full_c.style
        .highlight_min(subset=["MAPE (%)"], color="rgba(45,106,79,0.4)")
        .format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE (%)": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

with col_short:
    st.markdown(f"### 📅 Short 5 ปี ({n_short_m} เดือน)")
    best_s = metrics_short_c.iloc[0]
    st.metric("🏆 Best Model", best_s["method"])
    st.metric("Best MAPE", f"{best_s['MAPE (%)']:.2f}%")
    st.dataframe(
        metrics_short_c.style
        .highlight_min(subset=["MAPE (%)"], color="rgba(45,106,79,0.4)")
        .format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "MAPE (%)": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

# ── Grouped bar chart: MAPE comparison ──
st.markdown("### 📊 MAPE Comparison: 10 ปี vs 5 ปี")

# Align both metric tables by method
all_methods_c = list(dict.fromkeys(
    metrics_full_c["method"].tolist() + metrics_short_c["method"].tolist()
))
mape_full_vals, mape_short_vals = [], []
for m in all_methods_c:
    r_f = metrics_full_c[metrics_full_c["method"] == m]
    r_s = metrics_short_c[metrics_short_c["method"] == m]
    mape_full_vals.append(round(r_f["MAPE (%)"].values[0], 2) if len(r_f) else None)
    mape_short_vals.append(round(r_s["MAPE (%)"].values[0], 2) if len(r_s) else None)

compare_bar_opt = {
    "tooltip": {"trigger": "axis"},
    "legend": {"data": ["10 ปี (Full)", "5 ปี (Short)"], "textStyle": {"color": "#CCC"}},
    "xAxis": {"type": "category", "data": all_methods_c, "axisLabel": {"rotate": 15}},
    "yAxis": {"type": "value", "name": "MAPE (%)", "axisLabel": {"formatter": "{value}%"}},
    "series": [
        {
            "name": "10 ปี (Full)", "type": "bar",
            "data": mape_full_vals,
            "itemStyle": {"color": "#FFD700", "borderRadius": [6, 6, 0, 0]},
            "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 11},
        },
        {
            "name": "5 ปี (Short)", "type": "bar",
            "data": mape_short_vals,
            "itemStyle": {"color": "#4ECDC4", "borderRadius": [6, 6, 0, 0]},
            "label": {"show": True, "position": "top", "formatter": "{c}%", "fontSize": 11},
        },
    ],
    "backgroundColor": "transparent",
}
st_echarts(options=compare_bar_opt, height="420px")

# ── Difference table ──
st.markdown("### 📋 ตารางเปรียบเทียบ MAPE รายวิธี")
import pandas as pd
compare_rows = []
for m in all_methods_c:
    r_f = metrics_full_c[metrics_full_c["method"] == m]
    r_s = metrics_short_c[metrics_short_c["method"] == m]
    mf = r_f["MAPE (%)"].values[0] if len(r_f) else None
    ms = r_s["MAPE (%)"].values[0] if len(r_s) else None
    diff = (ms - mf) if mf is not None and ms is not None else None
    winner = "Full 10yr" if diff and diff > 0 else ("Short 5yr" if diff and diff < 0 else "เท่ากัน")
    compare_rows.append({
        "วิธี": m,
        "MAPE 10yr (%)": mf,
        "MAPE 5yr (%)": ms,
        "Δ MAPE (%)": round(diff, 2) if diff is not None else None,
        "ดีกว่า": winner,
    })
compare_df = pd.DataFrame(compare_rows)
st.dataframe(compare_df.style.format({
    "MAPE 10yr (%)": "{:.2f}", "MAPE 5yr (%)": "{:.2f}", "Δ MAPE (%)": "{:+.2f}",
}), use_container_width=True, hide_index=True)

# ── Conclusion ──
better_label = "10 ปี (Full)" if best_f["MAPE (%)"] <= best_s["MAPE (%)"] else "5 ปี (Short)"
st.markdown(f"""
<div class="highlight-box">
<b>🔍 สรุปการเปรียบเทียบ:</b><br><br>
• ข้อมูล <b>10 ปี</b> → Best MAPE = <b>{best_f['MAPE (%)']:.2f}%</b> ({best_f['method']})<br>
• ข้อมูล <b>5 ปี</b> → Best MAPE = <b>{best_s['MAPE (%)']:.2f}%</b> ({best_s['method']})<br><br>
✅ <b>ชุดข้อมูลที่ให้ผลดีกว่า: {better_label}</b><br><br>
<b>ข้อสังเกต:</b> ข้อมูลที่มากกว่าไม่ได้ดีกว่าเสมอไป — ถ้าข้อมูลเก่ามีลักษณะต่างจากปัจจุบันมาก
(เช่น ราคาทองคำก่อน 2020 อยู่ระดับ $1,200–1,800 แต่ปี 2024–2025 พุ่งเกิน $2,500)
การใช้เฉพาะข้อมูลล่าสุดอาจช่วยให้โมเดลจับ Pattern ปัจจุบันได้ดีกว่า
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────
st.markdown(f"""
<div class="section-box">
<h3>📌 สรุปภาพรวม</h3>

| หัวข้อ | รายละเอียด |
|--------|-----------|
| **ข้อมูลที่ใช้** | XAU/USD 2016–2025 (10 ปี, {n_full_m} เดือน) |
| **Metrics ที่ใช้** | MAE, RMSE, MAPE — มาตรฐานสากล Time Series Forecasting |
| **เกณฑ์ตัดสิน** | เรียงลำดับจาก MAPE ต่ำสุด |
| **แบบจำลองที่ชนะ (10 ปี)** | **{best_f['method']}** — MAPE = {best_f['MAPE (%)']:.2f}% |
| **แบบจำลองที่ชนะ (5 ปี)** | **{best_s['method']}** — MAPE = {best_s['MAPE (%)']:.2f}% |
| **Test Period** | {test_months} เดือนสุดท้าย |
| **ชุดข้อมูลที่ดีกว่า** | **{better_label}** |
| **ข้อสังเกต** | ข้อมูลทองคำมี Strong Uptrend → วิธีที่จับ Trend ได้ดีจะชนะ |
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("หน้า 6/7 — สรุปรูปแบบที่ดีที่สุด")
