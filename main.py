import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize

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

# ===== สร้างขนาดแปลงเฉลี่ยตามเกรด =====
df['%บ้านเดี่ยว'] = df['บ้านเดี่ยว'] / df['จำนวนหลัง'].replace(0, 1)
df['%บ้านแฝด'] = df['บ้านแฝด'] / df['จำนวนหลัง'].replace(0, 1)
df['%ทาวโฮม'] = df['ทาวโฮม'] / df['จำนวนหลัง'].replace(0, 1)

# ป้องกันค่าผิดปกติหรือ division by zero
df['ขนาดแปลงทาวน์โฮม'] = df.apply(lambda row: (row['พื้นที่จัดจำหน่าย(ตรม)'] * row['%ทาวโฮม']) / row['ทาวโฮม'] if row['ทาวโฮม'] > 0 else None, axis=1)
df['ขนาดแปลงบ้านแฝด'] = df.apply(lambda row: (row['พื้นที่จัดจำหน่าย(ตรม)'] * row['%บ้านแฝด']) / row['บ้านแฝด'] if row['บ้านแฝด'] > 0 else None, axis=1)
df['ขนาดแปลงบ้านเดี่ยว'] = df.apply(lambda row: (row['พื้นที่จัดจำหน่าย(ตรม)'] * row['%บ้านเดี่ยว']) / row['บ้านเดี่ยว'] if row['บ้านเดี่ยว'] > 0 else None, axis=1)

# คำนวณเฉลี่ยแยกตามเกรด
avg_sizes = df.groupby('เกรดโครงการ')[['ขนาดแปลงทาวน์โฮม', 'ขนาดแปลงบ้านแฝด', 'ขนาดแปลงบ้านเดี่ยว']].mean().to_dict()

# ===== ฟังก์ชันใหม่: ปรับจำนวนแปลงให้แม่นยำกับพื้นที่ขาย =====
def balance_units_by_area(พื้นที่ขาย_เป้าหมาย, ขนาด_T, ขนาด_F, ขนาด_D, x0=[80, 25, 40]):
    def objective(x):
        TH, F, D = x
        พท = (TH * ขนาด_T + F * ขนาด_F + D * ขนาด_D) * 4  # ตารางเมตร
        return abs(พท - พื้นที่ขาย_เป้าหมาย)

    bounds = [(0, None), (0, None), (0, None)]
    result = minimize(objective, x0, bounds=bounds)
    TH, F, D = map(round, result.x)
    return TH, F, D

# ===== ตัวอย่างการเรียกใช้ =====
# พท_ขาย = 32000  # พื้นที่จัดจำหน่ายที่พยากรณ์ได้
# เกรด = "PRIMO"  # ตัวอย่างเกรด
# TH, F, D = balance_units_by_area(
#     พท_ขาย,
#     avg_sizes['ขนาดแปลงทาวน์โฮม'].get(เกรด, 38.3),
#     avg_sizes['ขนาดแปลงบ้านแฝด'].get(เกรด, 53.19),
#     avg_sizes['ขนาดแปลงบ้านเดี่ยว'].get(เกรด, 62.44)
# )
# st.write("ทาวน์โฮม:", TH, "บ้านแฝด:", F, "บ้านเดี่ยว:", D)


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
