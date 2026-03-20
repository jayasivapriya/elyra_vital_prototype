import streamlit as st
import cv2
import numpy as np
import folium
from streamlit_folium import folium_static

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(page_title="Elyra Vital", page_icon="🌸", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; font-weight: bold; }
    .stHeader { color: #2e4053; }
    .diag-box { padding: 20px; border-radius: 10px; margin-bottom: 10px; border-left: 8px solid; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CORE ENGINEERING LOGIC ---

def analyze_ocular_advanced(image_file):
    """Improved and stable ocular analysis"""

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    raw_img = cv2.imdecode(file_bytes, 1)

    # --- ROI (focus center area - approx eye region) ---
    h, w, _ = raw_img.shape
    roi = raw_img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

    # --- 1. HSV MASK (STRICT YELLOW DETECTION) ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_jaundice = np.array([20, 80, 120])
    upper_jaundice = np.array([35, 255, 255])

    mask_jaundice = cv2.inRange(hsv, lower_jaundice, upper_jaundice)

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    mask_jaundice = cv2.morphologyEx(mask_jaundice, cv2.MORPH_OPEN, kernel)

    mask_pixels = np.sum(mask_jaundice > 0)

    if mask_pixels > 5000:
        blue_val = cv2.mean(roi[:,:,0], mask=mask_jaundice)[0]
        green_val = cv2.mean(roi[:,:,1], mask=mask_jaundice)[0]
        red_val = cv2.mean(roi[:,:,2], mask=mask_jaundice)[0]

        avg_rg = (red_val + green_val) / 2
        by_ratio = blue_val / avg_rg if avg_rg > 0 else 1.0
    else:
        by_ratio = 1.0

    # --- 2. ANEMIA CALCULATION (COLOR BALANCED) ---
    img_float = roi.astype(float)
    avg_total = np.mean(img_float)

    for i in range(3):
        avg_chan = np.mean(img_float[:,:,i])
        if avg_chan > 0:
            img_float[:,:,i] *= (avg_total / avg_chan)

    balanced_img = np.clip(img_float, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(balanced_img, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)
    avg_a = np.mean(a)

    # --- Vessel density ---
    gray = cv2.cvtColor(balanced_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    vessel_density = (np.sum(edges > 0) / edges.size) * 100

    return avg_a, vessel_density, by_ratio


def analyze_hair_density(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    h, w, _ = img.shape

    # Focus center region
    roi = img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Smooth
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # 🔥 Detect edges (hair-like lines)
    edges = cv2.Canny(gray, 50, 150)

    # 🔥 Morphology to keep only thin lines
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    hair_pixels = np.sum(cleaned > 0)
    total_pixels = cleaned.size

    density_ratio = hair_pixels / total_pixels

    # 🔥 Balanced scoring
    if density_ratio > 0.15:
        score = 4
    elif density_ratio > 0.10:
        score = 3
    elif density_ratio > 0.06:
        score = 2
    elif density_ratio > 0.03:
        score = 1
    else:
        score = 0

    return score, density_ratio
def get_action_engine_route(diagnosis):
    hospitals = [
        {"name": "Metro General Hospital", "pos": [12.9716, 80.2447], "spec": "General", "rank": 1},
        {"name": "Sree Gynaecology & PCOS Hub", "pos": [12.9850, 80.2550], "spec": "PCOS", "rank": 5},
        {"name": "Advanced Hematology Center", "pos": [12.9550, 80.2250], "spec": "Anemia", "rank": 5},
        {"name": "Liver & Hepatology Care", "pos": [12.9650, 80.2600], "spec": "Jaundice", "rank": 5}
    ]

    user_pos = [12.9716, 80.2447]
    best_h = None
    min_weight = float('inf')

    for h in hospitals:
        dist = ((h['pos'][0]-user_pos[0])**2 + (h['pos'][1]-user_pos[1])**2)**0.5
        spec_match = 5 if h['spec'] == diagnosis else 1
        weight = (dist * 1.2) + (1 / spec_match)

        if weight < min_weight:
            min_weight = weight
            best_h = h

    return best_h, hospitals


# --- 3. UI DASHBOARD ---

st.title("🌸 Elyra Vital: AI Women's Health Suite")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["👁️ Ocular Intelligence", "🧬 PCOS Intelligence", "📍 The Action Engine"])

if 'diagnosis' not in st.session_state:
    st.session_state['diagnosis'] = "General"

# --- TAB 1 ---
with tab1:
    st.header("Ocular Diagnostic Suite")

    input_method = st.radio("Select Input Source:", ["Live Camera", "Upload Image"])
    img_file = st.camera_input("Capture Scan") if input_method == "Live Camera" else st.file_uploader("Upload Image", type=['jpg','png','jpeg'])

    if img_file:
        a_val, density, by_ratio = analyze_ocular_advanced(img_file)

        st.markdown("### **Clinical Dashboard**")
        c1, c2, c3 = st.columns(3)

        # --- ANEMIA ---
        with c1:
            st.metric("Erythema Index (a*)", f"{a_val:.2f}")
            if a_val < 140:
                st.error("🚨 ANAEMIA DETECTED")
                st.session_state['diagnosis'] = "Anemia"
            else:
                st.success("✅ ANAEMIA: NORMAL")

        # --- VESSELS ---
        with c2:
            st.metric("Vessel Density", f"{density:.2f}%")
            if density < 2:
                st.warning("⚠️ Low Vessel Density")
            else:
                st.success("Healthy Vascularity")

        # --- JAUNDICE ---
        with c3:
            st.metric("Blue-to-Yellow Ratio", f"{by_ratio:.2f}")

            if by_ratio < 0.85:
                st.error("🚨 JAUNDICE DETECTED")
                st.markdown(
                    "<div style='background-color:#ffffcc; padding:15px; border-radius:10px; color:black;'>"
                    "<b>High Risk:</b> Strong scleral yellowing detected. Recommend bilirubin test."
                    "</div>",
                    unsafe_allow_html=True
                )
                st.session_state['diagnosis'] = "Jaundice"

            elif by_ratio < 0.92:
                st.warning("⚠️ MILD YELLOWING")

            else:
                st.success("✅ JAUNDICE: NORMAL")


# --- TAB 2 ---
# --- TAB 2 ---
with tab2:
    st.header("🧬 PCOS Intelligence")

    col1, col2, col3 = st.columns(3)

    with col1:
        lip_img = st.file_uploader("Upper Lip", type=['jpg','png','jpeg'])

    with col2:
        chin_img = st.file_uploader("Chin", type=['jpg','png','jpeg'])

    with col3:
        abd_img = st.file_uploader("Lower Abdomen", type=['jpg','png','jpeg'])

    # ===============================
    # 🧪 IMAGE ANALYSIS
    # ===============================
    if lip_img and chin_img and abd_img:

        s1, d1 = analyze_hair_density(lip_img)
        s2, d2 = analyze_hair_density(chin_img)
        s3, d3 = analyze_hair_density(abd_img)

        total_score = s1 + s2 + s3

        st.markdown("### 📊 Region-wise mFG Scores")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Upper Lip", f"{s1}/4")

        with c2:
            st.metric("Chin", f"{s2}/4")

        with c3:
            st.metric("Abdomen", f"{s3}/4")

        st.markdown("---")
        st.metric("🧮 Total mFG Score", f"{total_score}/12")

        # ===============================
# 🧠 PCOS DECISION (STABLE)
# ===============================

valid_regions = sum([1 for s in [s1, s2, s3] if s >= 2])

if total_score >= 8 and valid_regions >= 2:
    st.error("🚨 PCOS DETECTED")
    st.session_state['diagnosis'] = "PCOS"

elif total_score >= 6:
    st.warning("⚠️ Possible PCOS")

else:
    st.success("✅ No Strong Indicators")

    # ===============================
# 📅 PERIOD CYCLE TRACKER (RESTORED)
# ===============================
st.markdown("---")
st.subheader("📅 Period Cycle Tracker")

from datetime import date

if "cycle_dates" not in st.session_state:
    st.session_state.cycle_dates = []

new_date = st.date_input("Select Period Start Date", key="period_date")

colA, colB = st.columns(2)

with colA:
    if st.button("➕ Add Entry"):
        if new_date not in st.session_state.cycle_dates:
            st.session_state.cycle_dates.append(new_date)
            st.success("Date added!")

with colB:
    if st.button("🗑️ Clear History"):
        st.session_state.cycle_dates = []
        st.warning("History cleared")

# Show history
if len(st.session_state.cycle_dates) > 0:
    st.markdown("### 📖 Cycle History")
    for d in sorted(st.session_state.cycle_dates):
        st.write(f"🟣 {d}")

# Analysis
cycle_irregular = False

if len(st.session_state.cycle_dates) >= 2:
    dates = sorted(st.session_state.cycle_dates)

    cycles = []
    for i in range(1, len(dates)):
        cycles.append((dates[i] - dates[i-1]).days)

    avg_cycle = sum(cycles) / len(cycles)
    last_cycle = cycles[-1]

    st.metric("📊 Average Cycle Length", f"{avg_cycle:.1f} days")

    if last_cycle > 35:
        st.error("🚨 Delayed Cycle Detected")
        cycle_irregular = True
    elif last_cycle < 21:
        st.warning("⚠️ Short Cycle Detected")
        cycle_irregular = True
    else:
        st.success("✅ Cycle Regular")

# --- TAB 3 ---
with tab3:
    st.header("Dijkstra Action Engine")

    diag = st.session_state['diagnosis']
    st.info(f"Optimizing Healthcare Route for: **{diag}**")

    best, all_h = get_action_engine_route(diag)

    m = folium.Map(location=[12.9716, 80.2447], zoom_start=13)

    for h in all_h:
        icon_color = 'green' if h['name'] == best['name'] else 'blue'
        folium.Marker(h['pos'], popup=h['name'], icon=folium.Icon(color=icon_color)).add_to(m)

    folium_static(m)

    st.success(f"**Optimal Destination:** {best['name']} ({best['spec']} Specialist)")
