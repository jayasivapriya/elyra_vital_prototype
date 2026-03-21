import folium
from streamlit_folium import folium_static
import requests
import streamlit as st
import cv2
import numpy as np
import heapq
import math

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

# ✅ Resize to reduce memory load
    raw_img = cv2.resize(raw_img, (400, 400))

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
    file_bytes = np.asarray(bytearray(image_file.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 3. Laplacian (detect follicle points)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # 4. Threshold
    _, thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)

    # 5. Density calculation
    hair_points = np.sum(thresh > 0)
    total_pixels = thresh.size
    density = hair_points / total_pixels

    # 6. mFG scoring (CALIBRATED)
    if density > 0.08:
        score = 4
    elif density > 0.05:
        score = 3
    elif density > 0.03:
        score = 2
    elif density > 0.015:
        score = 1
    else:
        score = 0

    return score, density

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_action_engine_route(diagnosis):
    # ===============================
    # 🏥 HOSPITAL DATABASE
    # ===============================
    hospitals = {
        "User": {"pos": (12.9716, 80.2447), "spec": "General"},

        "Metro General Hospital": {
            "pos": (12.9716, 80.2447),
            "spec": "General"
        },

        "Sree Gynaecology & PCOS Hub": {
            "pos": (12.9850, 80.2550),
            "spec": "PCOS"
        },

        "Advanced Hematology Center": {
            "pos": (12.9550, 80.2250),
            "spec": "Anemia"
        },

        "Liver & Hepatology Care": {
            "pos": (12.9650, 80.2600),
            "spec": "Jaundice"
        }
    }

    # ===============================
    # 🧠 GRAPH BUILDING
    # ===============================
    graph = {}

    for h1 in hospitals:
        graph[h1] = {}
        for h2 in hospitals:
            if h1 != h2:
                dist = calculate_distance(hospitals[h1]["pos"], hospitals[h2]["pos"])

                # 🎯 Specialization priority
                spec_bonus = 0
                if hospitals[h2]["spec"] == diagnosis:
                    spec_bonus = -0.05   # lower weight → higher priority

                graph[h1][h2] = dist + spec_bonus

    # ===============================
    # 🚀 DIJKSTRA
    # ===============================
    start = "User"
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    pq = [(0, start)]
    previous = {}

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        for neighbor in graph[current_node]:
            weight = graph[current_node][neighbor]
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    # ===============================
    # 🎯 FIND BEST MATCH
    # ===============================
    best_hospital = None
    min_dist = float('inf')

    for h in hospitals:
        if h != "User" and hospitals[h]["spec"] == diagnosis:
            if distances[h] < min_dist:
                min_dist = distances[h]
                best_hospital = h

    # fallback if no exact match
    if best_hospital is None:
        best_hospital = min(
            [h for h in hospitals if h != "User"],
            key=lambda x: distances[x]
        )

    return best_hospital, hospitals
def find_nearest_hospital(lat, lon):
    API_KEY = "YOUR_API_KEY"  # 🔴 replace this

    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lon}&radius=5000&type=hospital&key={API_KEY}"

    response = requests.get(url)
    data = response.json()

    if data.get("results"):
        hospital = data["results"][0]
        name = hospital["name"]
        loc = hospital["geometry"]["location"]

        return name, loc["lat"], loc["lng"]
    else:
        return "No hospital found", lat, lon
# --- 3. UI DASHBOARD ---

st.title("🌸 Elyra Vital: AI Women's Health Suite")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["👁️ Ocular Intelligence", "🧬 PCOS Intelligence", "📍 The Action Engine"])

if 'diagnosis' not in st.session_state:
    st.session_state['diagnosis'] = "General"

# --- TAB 1 ---
# --- TAB 1 ---
with tab1:
    st.header("👁️ Ocular Intelligence")

    subtab1, subtab2 = st.tabs(["🩸 Anaemia Detection", "🟡 Jaundice Detection"])

    # ===============================
    # 🩸 ANAEMIA TAB
    # ===============================
    with subtab1:
        st.subheader("Palpebral Conjunctiva Analysis")

    input_method = st.radio(
        "Select Input Source:",
        ["Upload Image", "Live Camera"],
        key="anaemia_input"
    )

    img_file = (
        st.camera_input("Capture Eye Image")
        if input_method == "Live Camera"
        else st.file_uploader("Upload Eye Image", type=['jpg','png','jpeg'], key="anaemia_upload")
    )

    if img_file:
        a_val, density, _ = analyze_ocular_advanced(img_file)

        st.markdown("### 📊 Clinical Parameters")

        c1, c2 = st.columns(2)

        # 🩸 ERYTHEMA
        with c1:
            st.metric("Erythema Index (a*)", f"{a_val:.2f}")

            if a_val < 135 and density < 1.5:
                st.error("🚨 ANAEMIA DETECTED")
                st.session_state['diagnosis'] = "Anemia"

            elif a_val < 145 or density < 2:
                st.warning("⚠️ Possible Mild Anaemia")
                st.session_state['diagnosis'] = "General"

            else:
                st.success("✅ Normal")
                st.session_state['diagnosis'] = "General"

        # 🩸 VESSELS
        with c2:
            st.metric("Vessel Density", f"{density:.2f}%")

            if density < 2:
                st.warning("⚠️ Low Vessel Density")
            else:
                st.success("Healthy Vascularity")

    else:
        st.info("📌 Upload or capture an eye image to analyze Anaemia")
    # ===============================
    # 🟡 JAUNDICE TAB
    # ===============================
    with subtab2:
        st.subheader("Sclera (Eye White) Analysis")

        input_method_j = st.radio(
            "Select Input Source:",
            ["Upload Image", "Live Camera"],
            key="jaundice_input"
        )

        img_file_j = (
            st.camera_input("Capture Eye Image")
            if input_method_j == "Live Camera"
            else st.file_uploader("Upload Eye Image", type=['jpg','png','jpeg'], key="jaundice_upload")
        )

        if img_file_j:
            _, _, by_ratio = analyze_ocular_advanced(img_file_j)

            st.markdown("### 📊 Clinical Parameters")

            st.metric("Blue-to-Yellow Ratio", f"{by_ratio:.2f}")

            if by_ratio < 0.85:
                st.error("🚨 JAUNDICE DETECTED")
                st.session_state['diagnosis'] = "Jaundice"

                st.markdown(
                    "<div style='background-color:#ffffcc; padding:15px; border-radius:10px; color:black;'>"
                    "<b>High Risk:</b> Strong scleral yellowing detected. Recommend bilirubin test."
                    "</div>",
                    unsafe_allow_html=True
                )

            elif by_ratio < 0.92:
                st.warning("⚠️ Mild Yellowing")

            else:
                st.success("✅ Normal")


# --- TAB 2 ---
with tab2:
    st.header("🧬 PCOS Intelligence")

    # ===============================
    # 📤 IMAGE INPUT
    # ===============================
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

        # PCOS Decision Logic (FIXED + STABLE)
        valid_regions = sum([1 for s in [s1, s2, s3] if s >= 2])

        if total_score >= 9 and valid_regions >= 2:
            st.error("🚨 PCOS DETECTED")
            st.session_state['diagnosis'] = "PCOS"

        elif total_score >= 6 and valid_regions >= 2:
            st.warning("⚠️ Possible PCOS")

        else:
            st.success("✅ No Strong Indicators")

    else:
        st.info("📌 Upload all 3 images to analyze PCOS")

    # ===============================
    # 📅 PERIOD CYCLE TRACKER
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

    # ===============================
    # 📊 LAST 3 CYCLE ANALYSIS
    # ===============================
    if len(st.session_state.cycle_dates) >= 3:

        dates = sorted(st.session_state.cycle_dates)
        last_three = dates[-3:]

        cycle1 = (last_three[1] - last_three[0]).days
        cycle2 = (last_three[2] - last_three[1]).days

        avg_cycle = (cycle1 + cycle2) / 2

        st.markdown("---")
        st.metric("📊 Average Cycle Length", f"{avg_cycle:.1f} days")

        if avg_cycle > 35:
            st.error("🚨 Period Delay Detected")

            if st.session_state.get('diagnosis') == "PCOS":
                st.error("⚠️ High Confidence PCOS (mFG + Irregular Cycles)")

        elif avg_cycle < 21:
            st.warning("⚠️ Short Cycle Detected")

        else:
            st.success("✅ Cycle Regular")

    elif len(st.session_state.cycle_dates) > 0:
        st.info("📌 Add at least 3 entries for cycle analysis")
# --- TAB 3 ---
with tab3:
    st.header("📍 Smart Healthcare Routing")

    st.subheader("📍 Enter Your Location")

    user_lat = st.number_input("Enter your Latitude", value=12.9716)
    user_lon = st.number_input("Enter your Longitude", value=80.2447)

    name, dest_lat, dest_lon = find_nearest_hospital(user_lat, user_lon)

    st.success(f"🏥 Nearest Hospital: **{name}**")

    import folium
    from streamlit_folium import folium_static

    m = folium.Map(location=[user_lat, user_lon], zoom_start=13)

    folium.Marker([user_lat, user_lon], tooltip="You").add_to(m)
    folium.Marker([dest_lat, dest_lon], tooltip=name).add_to(m)

    folium.PolyLine([[user_lat, user_lon], [dest_lat, dest_lon]]).add_to(m)

    folium_static(m)
