import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Smart Layout AI", page_icon="📇️", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f4f8;
    }
    .stButton>button {
        background: linear-gradient(to right, #0f4c75, #3282b8);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 0.6em 2em;
    }
    div[data-testid="metric-container"] {
        background-color: white;
        border-radius: 12px;
        padding: 1em;
        margin: 10px 0;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="metric-container"] > label, div[data-testid="metric-container"] > div {
        color: #1f2937 !important;
        font-weight: 600;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

df = pd.read_excel("datalayout.xlsx")
df.columns = df.columns.str.strip()

# แสดง Map ให้ผู้ใช้เลือกโครงการใกล้เคียง
st.markdown("## 🗺️ เลือกตำแหน่งโครงการใกล้เคียงบนแผนที่")
m = folium.Map(location=[13.736717, 100.523186], zoom_start=6)
marker_data = []

for i, row in df.iterrows():
    if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
        popup = f"{row['ชื่อโครงการ']} ({row['จังหวัด']}, {row['เกรดโครงการ']})"
        folium.Marker([row['lat'], row['lon']], popup=popup).add_to(m)
        marker_data.append((row['ชื่อโครงการ'], row['lat'], row['lon']))

selected = st_folium(m, width=700, height=450)

if selected and selected['last_clicked']:
    lat_clicked = selected['last_clicked']['lat']
    lon_clicked = selected['last_clicked']['lng']
    st.success(f"คุณเลือกตำแหน่งใกล้: lat={lat_clicked:.4f}, lon={lon_clicked:.4f}")
    
    # หาจังหวัดหรือโครงการที่ใกล้เคียงที่สุด
    df['ระยะทาง'] = ((df['lat'] - lat_clicked)**2 + (df['lon'] - lon_clicked)**2)**0.5
    nearest = df.sort_values('ระยะทาง').iloc[0]
    st.info(f"โครงการใกล้เคียงที่สุด: {nearest['ชื่อโครงการ']} ({nearest['จังหวัด']}, {nearest['เกรดโครงการ']})")

    # สามารถใช้ข้อมูล nearest เพื่อกำหนด input prediction ได้ เช่น:
    # จังหวัด = nearest['จังหวัด']
    # เกรด = nearest['เกรดโครงการ']
    # รูปร่าง = nearest['รูปร่างที่ดิน']
    # แล้วไปต่อในโมเดลต่อไป
else:
    st.warning("กรุณาเลือกตำแหน่งจากแผนที่เพื่อดำเนินการต่อ")

# ====== FORM ======
st.markdown("## 📋 กรอกข้อมูลโครงการ")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        จังหวัด = st.selectbox("📍 จังหวัด", sorted(df['จังหวัด'].dropna().unique()))
        รูปร่าง = st.selectbox("🧱️ รูปร่างที่ดิน", sorted(df['รูปร่างที่ดิน'].dropna().unique()))
    with col2:
        เกรด = st.selectbox("🏧 เกรดโครงการ", sorted(df['เกรดโครงการ'].dropna().unique()))
        พื้นที่_วา = st.number_input("📀 พื้นที่โครงการ (ตารางวา)", min_value=250, value=7500, step=100)
    submitted = st.form_submit_button("🚀 เริ่มพยากรณ์")

# ====== PREDICT ======
if submitted:
    area = พื้นที่_วา * 4
    rai = area / 1600
    input_df = pd.DataFrame([{ 'จังหวัด': จังหวัด, 'เกรดโครงการ': เกรด, 'พื้นที่โครงการ(ตรม)': area, 'รูปร่างที่ดิน': รูปร่าง }])
    encoded = pd.get_dummies(input_df)
    for col in X.columns:
        if col not in encoded.columns:
            encoded[col] = 0
    encoded = encoded[X.columns]
    pred = model.predict(encoded)[0]

    พท_สาธา = pred[0] * area
    พท_ขาย = pred[1] * area
    พท_สวน = pred[2] * area
    พท_ถนน = พท_สาธา * ถนน_ต่อ_สาธารณะ_เฉลี่ย  # ✅ ใช้สัดส่วนจากข้อมูลอดีต
    หลังรวม = pred[3] * rai

    ratio_hist = get_ratio_from_lookup(เกรด, area)
    if ratio_hist:
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in ratio_hist]
    else:
        total = sum(pred[4:8]) or 1
        raw_ratios = [r / total for r in pred[4:8]]
        raw_ratios = adjust_by_grade_policy(เกรด, raw_ratios)
        ทาวโฮม, บ้านแฝด, บ้านเดี่ยว, อาคารพาณิชย์ = [หลังรวม * r for r in raw_ratios]

    ซอย = หลังรวม / avg_ซอยต่อหลัง.get(เกรด, 12)

    st.markdown("---")
    st.markdown("## 🌟 ผลลัพธ์พยากรณ์")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("พื้นที่สาธารณะ", f"{พท_สาธา / 4:,.0f} ตร.วา")
        st.metric("พื้นที่จัดจำหน่าย", f"{พท_ขาย / 4:,.0f} ตร.วา")
        st.metric("พื้นที่สวน", f"{พท_สวน / 4:,.0f} ตร.วา")
        st.metric("พื้นที่ถนน", f"{พท_ถนน / 4:,.0f} ตร.วา")
    with col2:
        st.metric("จำนวนแปลงรวม", f"{หลังรวม:,.0f} หลัง")
        st.metric("จำนวนซอย", f"{ซอย:,.0f} ซอย")

    st.markdown("### 🏡 แยกตามประเภทบ้าน")
    st.markdown(f"""
        - ทาวน์โฮม: **{ทาวโฮม:,.0f}** หลัง  
        - บ้านแฝด: **{บ้านแฝด:,.0f}** หลัง  
        - บ้านเดี่ยว: **{บ้านเดี่ยว:,.0f}** หลัง  
        - อาคารพาณิชย์: **{อาคารพาณิชย์:,.0f}** หลัง  
    """)

    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    st.markdown("### 📈 ความแม่นยำของโมเดล (Train Set)")
    st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

st.markdown("---")
st.caption("Developed by mmethaa | Smart Layout AI 🚀")
