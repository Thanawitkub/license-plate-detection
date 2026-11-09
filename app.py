import os
import re
import time
import tempfile
import difflib
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import torch
import streamlit as st
from ultralytics import YOLO
import easyocr

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ==============================================================================
# 1. การตั้งค่าหน้าเว็บและธีมดีไซน์ (CCTV Dark Cyberpunk Surveillance Theme)
# ==============================================================================
st.set_page_config(
    page_title="AI CCTV - ระบบตรวจจับและอ่านป้ายทะเบียนอัจฉริยะ",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับสไตล์ห้องควบคุม CCTV แบบ Dark Cyberpunk
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Theme */
    html, body, .stApp {
        font-family: 'Chakra Petch', sans-serif !important;
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0b0f19;
    }

    /* Top CCTV Header Bar */
    .cctv-header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(90deg, #111827 0%, #0d1522 100%);
        border-left: 6px solid #10b981;
        border-top: 1px solid #1f2937;
        border-right: 1px solid #1f2937;
        border-bottom: 1px solid #1f2937;
        padding: 14px 24px;
        border-radius: 8px;
        margin-bottom: 22px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }
    .cctv-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .cctv-title-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 400;
    }

    /* Blinking Live Indicator */
    @keyframes live-blink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.25; transform: scale(0.9); }
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.5);
        padding: 6px 14px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .live-dot {
        width: 10px;
        height: 10px;
        background-color: #ef4444;
        border-radius: 50%;
        box-shadow: 0 0 10px #ef4444;
        animation: live-blink 1.2s infinite ease-in-out;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d121f;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1e293b;
    }

    /* Hardware Status Card */
    .hw-status-card {
        padding: 10px 14px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 15px;
    }
    .hw-cuda {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .hw-cpu {
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }

    /* Right Column: Counter & History Cards */
    .summary-counter-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .summary-counter-label {
        font-size: 0.82rem;
        color: #94a3b8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .summary-counter-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }

    .history-card-container {
        background: #111827;
        border: 1px solid #1e293b;
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .history-card-container:hover {
        border-color: #38bdf8;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2);
    }
    .neon-plate-text {
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
    }
    .history-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
    }
    .history-conf-badge {
        font-size: 0.72rem;
        font-family: 'JetBrains Mono', monospace;
        color: #10b981;
        margin-top: 3px;
    }

    /* Buttons Styling */
    .stButton>button {
        font-family: 'Chakra Petch', sans-serif !important;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    /* Tab bar customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 600;
        color: #94a3b8;
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #10b981 !important;
        border-bottom: 2px solid #10b981 !important;
        background: rgba(16, 185, 129, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Bar สไตล์จอ CCTV
st.markdown(
    """
    <div class="cctv-header-bar">
        <div>
            <div class="cctv-title">📹 SMART CCTV LICENSE PLATE RECOGNITION (ANPR)</div>
            <div class="cctv-title-sub">AUTOMATED VEHICLE IDENTIFICATION & SECURITY MONITORING SYSTEM</div>
        </div>
        <div>
            <div class="live-badge">
                <span class="live-dot"></span>
                LIVE SYSTEM
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 2. การโหลดโมเดล AI (Caching & Multi-Hardware)
# ==============================================================================
is_cuda_available = torch.cuda.is_available()

@st.cache_resource(show_spinner="กำลังโหลดโมเดลตรวจจับป้ายทะเบียน YOLOv8...")
def load_yolo_model(weights_path: str = "best.pt") -> YOLO:
    """โหลดโมเดล YOLOv8 จากไฟล์ weights พร้อม Caching"""
    if not os.path.exists(weights_path):
        st.error(f"❌ ไม่พบไฟล์โมเดล '{weights_path}' ในไดเรกทอรีปัจจุบัน!")
        st.stop()
    model = YOLO(weights_path)
    return model

@st.cache_resource(show_spinner="กำลังโหลดโมเดลอ่านตัวอักษร EasyOCR...")
def load_easyocr_reader(languages_tuple: tuple, use_gpu: bool) -> easyocr.Reader:
    """โหลดโมเดล EasyOCR รองรับการสลับ GPU / CPU อัตโนมัติ พร้อม Caching"""
    return easyocr.Reader(list(languages_tuple), gpu=use_gpu)


# ==============================================================================
# 3. ฟังก์ชัน Image Preprocessing & Text Cleaning
# ==============================================================================
def preprocess_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    ปรับแต่งภาพป้ายทะเบียนก่อนส่งเข้าโมเดล OCR:
    1. หากป้ายเล็กกว่า 40x100 พิกเซล ให้ขยายขนาดภาพด้วย cv2.INTER_CUBIC
    2. แปลงภาพเป็น Grayscale
    3. เร่งความคมชัดของตัวอักษรด้วย CLAHE (clipLimit=2.0, tileGridSize=(8, 8))
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    # ขยายขนาดภาพหากขนาดเล็กเกินไป
    if h < 40 or w < 100:
        scale = max(40.0 / max(h, 1), 100.0 / max(w, 1), 2.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # แปลงเป็นภาพ Grayscale
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # เร่งความคมชัดด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced

def clean_plate_text(text: str) -> str:
    """
    ทำความสะอาดข้อความป้ายทะเบียน:
    ตัดอักขระพิเศษที่ไม่ต้องการออก โดยใช้ Regex กรองเฉพาะภาษาไทย ภาษาอังกฤษ ตัวเลข และช่องว่าง
    """
    if not text:
        return ""
    # กรองเฉพาะ [a-zA-Z0-9_], ช่องว่าง และภาษาไทย (\u0E00-\u0E7F)
    cleaned = re.sub(r'[^\w\s\u0E00-\u0E7F]', '', text)
    # รวมช่องว่างซ้ำซ้อนให้เหลือช่องว่างเดียวและตัดขอบ
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ==============================================================================
# Helper Functions: Drawing Bounding Boxes & OSD Overlays
# ==============================================================================
def draw_thai_text_on_bgr(img_bgr: np.ndarray, text: str, org: tuple, font_size: int = 18,
                          text_color: tuple = (255, 255, 255), bg_color: tuple = None) -> np.ndarray:
    """
    วาดข้อความภาษาไทย/อังกฤษลงบนภาพ BGR ด้วย PIL หรือ fallback เป็น cv2.putText
    """
    if not PIL_AVAILABLE:
        cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
        return img_bgr

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # ค้นหาฟอนต์ที่รองรับภาษาไทยในระบบ
    font = None
    candidate_fonts = [
        "C:/Windows/Fonts/leelawad.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/angsana.ttf",
        "C:/Windows/Fonts/cordia.ttf",
        "leelawad.ttf",
        "tahoma.ttf"
    ]
    for fp in candidate_fonts:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    x, y = org
    if bg_color is not None:
        bbox = draw.textbbox((x, y), text, font=font)
        # เผื่อ padding รอบข้อความ
        padded_bbox = (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2)
        draw.rectangle(padded_bbox, fill=bg_color)

    draw.text((x, y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_cctv_bounding_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                            label: str = "", conf: float = None) -> np.ndarray:
    """
    วาด Bounding Box สไตล์ Cyber CCTV:
    - กรอบสีเขียวสะท้อนแสง (0, 255, 127) หนา 2px
    - มุมไฮไลท์ 4 มุมสไตล์กล้อง CCTV ด้วยเส้นสีเหลืองนีออน (0, 255, 255)
    - ป้ายข้อความพร้อมค่า Confidence
    """
    box_color = (0, 255, 127)      # เขียวสะท้อนแสง
    corner_color = (0, 255, 255)   # เหลืองนีออน (ใน BGR คือ B=0, G=255, R=255)

    # 1. วาดกรอบสี่เหลี่ยมหลัก
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # 2. วาดมุมไฮไลท์ 4 มุม
    box_w = x2 - x1
    box_h = y2 - y1
    corner_len = min(16, max(6, box_w // 4), max(6, box_h // 4))
    thickness = 3

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), corner_color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), corner_color, thickness)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), corner_color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), corner_color, thickness)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), corner_color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), corner_color, thickness)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), corner_color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), corner_color, thickness)

    # 3. วาดป้ายข้อความ
    display_text = label
    if conf is not None:
        display_text = f"{display_text} [{conf*100:.0f}%]" if display_text else f"PLATE [{conf*100:.0f}%]"

    if display_text:
        text_y = max(10, y1 - 24)
        frame = draw_thai_text_on_bgr(
            img_bgr=frame,
            text=display_text,
            org=(x1, text_y),
            font_size=18,
            text_color=(255, 255, 255),
            bg_color=(15, 23, 42)
        )

    return frame

def draw_osd_overlay(frame: np.ndarray, actual_fps: float) -> np.ndarray:
    """
    วาดข้อมูลกล้องวงจรปิดมุมบนซ้าย (OSD Overlay):
    `CAM-01 [GATE] REC [o] {Timestamp} FPS: {Actual_FPS}`
    """
    h, w = frame.shape[:2]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    osd_text = f"CAM-01 [GATE]  REC [o]  {now_str}  FPS: {actual_fps:.1f}"

    # วาดแถบพื้นหลังสีเข้มโปร่งแสง
    overlay = frame.copy()
    bar_width = min(w - 16, 560)
    cv2.rectangle(overlay, (12, 10), (bar_width, 42), (11, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ขอบแถบ OSD สีเขียวนีออน
    cv2.rectangle(frame, (12, 10), (bar_width, 42), (16, 185, 129), 1)

    # ข้อความ OSD
    cv2.putText(frame, osd_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (226, 232, 240), 1, cv2.LINE_AA)
    # จุด REC สีแดง
    rec_pos_x = 188
    cv2.circle(frame, (rec_pos_x, 27), 4, (0, 0, 255), -1)

    return frame

def cap_resolution_720p(image: np.ndarray) -> np.ndarray:
    """หากภาพมีความละเอียดเกิน 720p ให้ย่อขนาดลงมาที่ 720p อัตโนมัติ เพื่อป้องกันคอขวด WebSocket"""
    if image is None:
        return image
    h, w = image.shape[:2]
    if h > 720 or w > 1280:
        scale = min(1280.0 / w, 720.0 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


# ==============================================================================
# Session State Initialization
# ==============================================================================
if 'detected_records' not in st.session_state:
    st.session_state.detected_records = []

if 'detected_plates' not in st.session_state:
    # เก็บประวัติ {clean_key: last_timestamp} สำหรับคูลดาวน์ป้องกันป้ายซ้ำ
    st.session_state.detected_plates = {}

if 'video_running' not in st.session_state:
    st.session_state.video_running = False


# ==============================================================================
# 7. ระบบป้องกันป้ายซ้ำอัจฉริยะ (Fuzzy Matching Deduplication)
# ==============================================================================
def register_detected_plate(full_text: str, yolo_conf: float, ocr_conf: float,
                            crop_bgr: np.ndarray, cooldown_sec: float) -> tuple[bool, str]:
    """
    ระบบป้องกันป้ายซ้ำ:
    - ตัดช่องว่างออกทั้งหมด (clean_key)
    - เปรียบเทียบกับป้ายที่เคยตรวจพบในช่วงคูลดาวน์ด้วย SequenceMatcher ratio > 0.75
    - หากซ้ำ: ขยายเวลาคูลดาวน์ และไม่เพิ่มประวัติซ้ำ
    - หากใหม่: เพิ่มลง st.session_state.detected_records
    คืนค่า (is_duplicate: bool, cleaned_text: str)
    """
    cleaned_text = clean_plate_text(full_text)
    clean_key = cleaned_text.replace(" ", "")

    # หากข้อความสั้นเกินไปถือว่าเป็น Noise
    if len(clean_key) < 2:
        return True, cleaned_text

    current_time = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ตรวจสอบกับประวัติที่มีอยู่
    for saved_key, last_seen in list(st.session_state.detected_plates.items()):
        if (current_time - last_seen) <= cooldown_sec:
            similarity = difflib.SequenceMatcher(None, clean_key, saved_key).ratio()
            if similarity > 0.75:
                # ป้ายเดิม: ขยายคูลดาวน์ออกไปตราบใดที่ยังอยู่ในมุมกล้อง
                st.session_state.detected_plates[saved_key] = current_time
                return True, cleaned_text

    # เป็นป้ายใหม่หรือพ้นช่วงคูลดาวน์แล้ว
    st.session_state.detected_plates[clean_key] = current_time

    # บันทึกภาพแบบ RGB เพื่อแสดงผล
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB) if crop_bgr is not None else None

    new_record = {
        'id': len(st.session_state.detected_records) + 1,
        'timestamp': now_str,
        'text': cleaned_text,
        'yolo_conf': float(yolo_conf),
        'ocr_conf': float(ocr_conf),
        'image': crop_rgb
    }
    st.session_state.detected_records.append(new_record)
    return False, cleaned_text


# ==============================================================================
# 4. Sidebar: แผงควบคุมระบบ (Surveillance Control)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🎛️ แผงควบคุมระบบ (SURVEILLANCE)")
    st.markdown("<hr style='margin: 8px 0 16px 0;'>", unsafe_allow_html=True)

    # กล่องข้อความแจ้งสถานะฮาร์ดแวร์
    if is_cuda_available:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "NVIDIA GPU"
        st.markdown(
            f"""
            <div class="hw-status-card hw-cuda">
                <span>ฮาร์ดแวร์: CUDA (GPU) 🚀</span>
                <span style="font-size: 0.75rem; color: #94a3b8;">{gpu_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="hw-status-card hw-cpu">
                <span>ฮาร์ดแวร์: CPU ⚙️</span>
                <span style="font-size: 0.75rem; color: #94a3b8;">STANDARD MODE</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 1. ตั้งค่า YOLO
    st.markdown("#### 🎯 ตั้งค่า YOLOv8")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=0.95, value=0.35, step=0.05)
    iou_threshold = st.slider("NMS IoU Threshold", min_value=0.20, max_value=0.90, value=0.45, step=0.05)

    st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)

    # 2. ตั้งค่า OCR
    st.markdown("#### 🔤 ตั้งค่า EasyOCR")
    ocr_lang_choice = st.selectbox(
        "ภาษาที่อ่าน",
        options=["th, en (ไทย & อังกฤษ)", "en (อังกฤษตัวเลขล้วน)"],
        index=0
    )
    selected_langs = ('th', 'en') if "th" in ocr_lang_choice else ('en',)
    ocr_conf_threshold = st.slider("OCR Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

    st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)

    # 3. ประสิทธิภาพการประมวลผล
    st.markdown("#### ⚡ ประสิทธิภาพการประมวลผล")
    cooldown_sec = st.slider("คูลดาวน์ป้องกันป้ายซ้ำ (วินาที)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    frame_skip = st.slider("ประมวลผล AI ทุกๆ N เฟรม (Frame Skip)", min_value=1, max_value=10, value=3, step=1)
    ocr_interval_sec = st.slider("ความถี่ในการรัน OCR ต่อป้าย (วินาที)", min_value=0.3, max_value=3.0, value=1.0, step=0.1)

    st.markdown("<hr style='margin: 14px 0;'>", unsafe_allow_html=True)

    # 4. ปุ่มล้างประวัติ
    if st.button("🗑️ ล้างประวัติป้ายที่ตรวจจับได้", use_container_width=True, type="secondary"):
        st.session_state.detected_records = []
        st.session_state.detected_plates = {}
        st.session_state.image_annotated_result = None
        st.session_state.image_detection_summary = None
        st.rerun()

    # 5. ส่งออกข้อมูล CSV (เฉพาะเมื่อมีข้อมูล)
    if len(st.session_state.detected_records) > 0:
        export_rows = []
        for r in st.session_state.detected_records:
            export_rows.append({
                "ลำดับ": r["id"],
                "วัน-เวลาที่ตรวจพบ": r["timestamp"],
                "ข้อความป้ายทะเบียน": r["text"],
                "ความมั่นใจ YOLO (%)": f"{r['yolo_conf'] * 100:.1f}%",
                "ความมั่นใจ OCR (%)": f"{r['ocr_conf'] * 100:.1f}%"
            })
        export_df = pd.DataFrame(export_rows)
        csv_bytes = export_df.to_csv(index=False).encode('utf-8-sig')

        st.download_button(
            label="📥 ดาวน์โหลดประวัติเป็น CSV",
            data=csv_bytes,
            file_name=f"cctv_license_plates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# โหลดโมเดล AI (เรียกใช้ Cache)
# ==============================================================================
yolo_model = load_yolo_model("best.pt")
ocr_reader = load_easyocr_reader(selected_langs, use_gpu=is_cuda_available)


# ==============================================================================
# แบ่งหน้าจอหลักออกเป็น 70:30 (col_main, col_results)
# ==============================================================================
col_main, col_results = st.columns([3, 1])


# ==============================================================================
# 5. แถบแสดงผลลัพธ์ประวัติทางขวา (30% - col_results)
# ==============================================================================
def render_detection_history(target_col):
    """
    ฟังก์ชันแสดงผลลัพธ์ประวัติป้ายทะเบียนล่าสุด 30 รายการ เรียงจากใหม่สุดไว้บนสุด
    แสดงผลในรูปแบบ Card สองคอลัมน์: ด้านซ้ายเป็นภาพป้ายทะเบียน, ด้านขวาเป็นรายละเอียด
    """
    total_count = len(st.session_state.detected_records)

    container = target_col.container()
    with container:
        # การ์ดสรุปจำนวน
        st.markdown(
            f"""
            <div class="summary-counter-card">
                <div class="summary-counter-label">จำนวนป้ายทั้งหมด</div>
                <div class="summary-counter-value">{total_count} คัน</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### 📋 บันทึกป้ายล่าสุด (History)")

        if total_count == 0:
            st.markdown(
                """
                <div style="background: #111827; border: 1px dashed #1e293b; border-radius: 6px; padding: 18px; text-align: center; color: #64748b; font-size: 0.85rem;">
                    ยังไม่พบป้ายทะเบียน<br>ระบบกำลังสแตนด์บายตรวจจับ
                </div>
                """,
                unsafe_allow_html=True
            )
            return

        # นำ 30 รายการล่าสุดมาแสดง โดยเรียงใหม่สุดไว้บนสุด
        latest_records = st.session_state.detected_records[::-1][:30]

        for item in latest_records:
            card_box = st.container()
            with card_box:
                c_img, c_info = st.columns([1, 1.3])
                with c_img:
                    if item['image'] is not None:
                        st.image(item['image'], use_container_width=True)
                    else:
                        st.markdown("<div style='background:#1f2937; height:45px; border-radius:4px;'></div>", unsafe_allow_html=True)
                with c_info:
                    st.markdown(
                        f"""
                        <div style="line-height: 1.35;">
                            <div class="neon-plate-text">{item['text'] if item['text'] else 'ไม่ระบุข้อความ'}</div>
                            <div class="history-meta">⏱️ {item['timestamp']}</div>
                            <div class="history-conf-badge">🎯 YOLO: {item['yolo_conf']*100:.0f}% | 🔤 OCR: {item['ocr_conf']*100:.0f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.markdown("<hr style='border:0; border-top: 1px solid #1e293b; margin: 6px 0 10px 0;'>", unsafe_allow_html=True)


# ==============================================================================
# 6. พื้นที่ทำงานหลักฝั่งซ้าย (70% - col_main): Tabs
# ==============================================================================
tab_video, tab_image = col_main.tabs(["🎥 โหมดวิดีโอ", "📸 โหมดรูปภาพ"])


# ------------------------------------------------------------------------------
# --- [แท็บที่ 1: โหมดวิดีโอ (Video Mode)] ---
# ------------------------------------------------------------------------------
with tab_video:
    video_file = st.file_uploader(
        "อัปโหลดไฟล์วิดีโอ CCTV (.mp4, .avi, .mov, .m4v)",
        type=["mp4", "avi", "mov", "m4v"],
        key="video_uploader"
    )

    col_btn_start, col_btn_stop = st.columns(2)
    start_video = col_btn_start.button("▶️ เริ่มตรวจจับ (Start)", key="start_vid_btn", use_container_width=True, type="primary")
    stop_video = col_btn_stop.button("⏹️ หยุด (Stop)", key="stop_vid_btn", use_container_width=True)

    if stop_video:
        st.session_state.video_running = False

    video_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_video and video_file is not None:
        st.session_state.video_running = True

        # บันทึกวิดีโอลง Temp file เพื่อให้ OpenCV เปิดอ่านได้
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.name)[1])
        tfile.write(video_file.read())
        tfile.flush()
        temp_video_path = tfile.name

        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("❌ ไม่สามารถเปิดไฟล์วิดีโอได้")
            st.session_state.video_running = False
        else:
            # Auto FPS & Resolution Cap
            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            fps_step = 2 if (orig_fps and orig_fps >= 45.0) else 1

            frame_idx = 0
            raw_frame_idx = 0
            prev_time = time.time()
            fps_smooth = orig_fps if (orig_fps and orig_fps > 0) else 25.0

            # แคชกล่องป้ายล่าสุดสำหรับ Strict Frame Skipping
            last_detected_boxes = []
            plate_ocr_tracker = {}  # {box_key: {'text': str, 'ocr_conf': float, 'last_ocr_time': float}}

            # Placeholder สำหรับอัปเดตพาเนลประวัติด้านขวาแบบ Realtime
            history_placeholder = col_results.empty()
            render_detection_history(history_placeholder)

            while cap.isOpened() and st.session_state.video_running:
                ret, raw_frame = cap.read()
                if not ret:
                    break

                raw_frame_idx += 1
                # Auto FPS Cap: หากเป็น 60 FPS ให้อ่านเฟรมเว้นเฟรม เพื่อให้เหลือ ~30 FPS
                if fps_step > 1 and (raw_frame_idx % fps_step != 0):
                    continue

                frame_idx += 1

                # คำนวณ Actual FPS
                curr_time = time.time()
                time_delta = curr_time - prev_time
                if time_delta > 0:
                    inst_fps = 1.0 / time_delta
                    fps_smooth = 0.88 * fps_smooth + 0.12 * inst_fps
                prev_time = curr_time

                # Auto Resolution Cap: ย่อขนาดลงมาที่ 720p อัตโนมัติหากเกิน
                display_frame = cap_resolution_720p(raw_frame)
                disp_h, disp_w = display_frame.shape[:2]

                # Strict Frame Skipping: รันโมเดล AI เฉพาะเมื่อ frame_idx % frame_skip == 0
                is_ai_frame = (frame_idx % frame_skip == 0)

                if is_ai_frame:
                    # Hard Resize for Inference: ย่อความกว้างเฟรมเหลือ 640px เสมอ
                    infer_w = 640
                    infer_h = max(int(disp_h * (infer_w / max(disp_w, 1))), 32)
                    infer_frame = cv2.resize(display_frame, (infer_w, infer_h))

                    # YOLO Inference
                    results = yolo_model.predict(
                        source=infer_frame,
                        imgsz=640,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )

                    new_boxes = []
                    scale_x = disp_w / infer_w
                    scale_y = disp_h / infer_h

                    for r in results:
                        for box in r.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf_val = float(box.conf[0].cpu().numpy())

                            # แปลงพิกัดกลับสู่สเกลของ display_frame
                            x1 = max(0, int(xyxy[0] * scale_x))
                            y1 = max(0, int(xyxy[1] * scale_y))
                            x2 = min(disp_w, int(xyxy[2] * scale_x))
                            y2 = min(disp_h, int(xyxy[3] * scale_y))

                            if (x2 - x1) < 10 or (y2 - y1) < 10:
                                continue

                            # ครอปป้ายทะเบียน
                            crop_bgr = display_frame[y1:y2, x1:x2].copy()

                            # ตรวจสอบความถี่ในการรัน OCR ต่อป้าย
                            center_key = f"{round(x1/40)*40}_{round(y1/40)*40}"
                            now_ts = time.time()

                            text_val = ""
                            ocr_conf_val = 0.0

                            # เช็คว่าเพิ่งรัน OCR สำหรับตำแหน่งนี้ไปหรือไม่
                            if center_key in plate_ocr_tracker and (now_ts - plate_ocr_tracker[center_key]['last_ocr_time'] < ocr_interval_sec):
                                text_val = plate_ocr_tracker[center_key]['text']
                                ocr_conf_val = plate_ocr_tracker[center_key]['ocr_conf']
                            else:
                                # Preprocessing สำหรับ OCR
                                preprocessed_crop = preprocess_for_ocr(crop_bgr)

                                # EasyOCR Reading
                                ocr_results = ocr_reader.readtext(preprocessed_crop)
                                recognized_words = []
                                conf_list = []

                                for res in ocr_results:
                                    _, ocr_word, ocr_prob = res
                                    if ocr_prob >= ocr_conf_threshold:
                                        recognized_words.append(ocr_word)
                                        conf_list.append(ocr_prob)

                                if recognized_words:
                                    raw_text = " ".join(recognized_words)
                                    ocr_conf_val = float(sum(conf_list) / len(conf_list))
                                    # ป้องกันป้ายซ้ำและบันทึก
                                    _, text_val = register_detected_plate(
                                        full_text=raw_text,
                                        yolo_conf=conf_val,
                                        ocr_conf=ocr_conf_val,
                                        crop_bgr=crop_bgr,
                                        cooldown_sec=cooldown_sec
                                    )
                                    plate_ocr_tracker[center_key] = {
                                        'text': text_val,
                                        'ocr_conf': ocr_conf_val,
                                        'last_ocr_time': now_ts
                                    }
                                    # อัปเดตแถบประวัติ
                                    render_detection_history(history_placeholder)

                            new_boxes.append({
                                'coords': (x1, y1, x2, y2),
                                'conf': conf_val,
                                'text': text_val,
                                'ocr_conf': ocr_conf_val
                            })

                    last_detected_boxes = new_boxes

                # วาด Bounding Box (เฟรมที่ข้ามจะคงตำแหน่งเดิมของ last_detected_boxes ไว้)
                for b_info in last_detected_boxes:
                    bx1, by1, bx2, by2 = b_info['coords']
                    b_conf = b_info['conf']
                    b_text = b_info['text']
                    display_frame = draw_cctv_bounding_box(
                        frame=display_frame,
                        x1=bx1, y1=by1, x2=bx2, y2=by2,
                        label=b_text,
                        conf=b_conf
                    )

                # วาด OSD Overlay
                display_frame = draw_osd_overlay(display_frame, fps_smooth)

                # แปลง BGR -> RGB ส่งขึ้นจอทันที
                display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            cap.release()
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

            if not st.session_state.video_running:
                status_placeholder.info("⏹️ ผู้ใช้กดหยุดการตรวจจับวิดีโอ")
            else:
                status_placeholder.success("✅ การตรวจจับวิดีโอเสร็จสิ้นสมบูรณ์")
                st.session_state.video_running = False


# ------------------------------------------------------------------------------
# --- [แท็บที่ 2: โหมดรูปภาพ (Image Mode)] ---
# ------------------------------------------------------------------------------
with tab_image:
    image_file = st.file_uploader(
        "อัปโหลดรูปภาพที่ต้องการตรวจจับป้ายทะเบียน (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    # ตรวจสอบว่าผู้ใช้อัปโหลดรูปใหม่หรือไม่ เพื่อรีเซ็ตผลลัพธ์เก่า
    if "last_uploaded_img_name" not in st.session_state:
        st.session_state.last_uploaded_img_name = None

    if image_file is not None:
        if st.session_state.last_uploaded_img_name != image_file.name:
            st.session_state.last_uploaded_img_name = image_file.name
            st.session_state.image_annotated_result = None
            st.session_state.image_detection_summary = None

        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            # Auto Resolution Cap 720p
            img_bgr = cap_resolution_720p(img_bgr)
            disp_h, disp_w = img_bgr.shape[:2]

            btn_detect_img = st.button("🔍 เริ่มตรวจจับป้ายทะเบียน (Detect)", key="detect_img_btn", type="primary")

            if btn_detect_img:
                with st.spinner("กำลังประมวลผลโมเดล AI (YOLO + EasyOCR)..."):
                    t_start = time.time()

                    # 1. Hard Resize for Inference: ย่อความกว้างเหลือ 640px
                    infer_w = 640
                    infer_h = max(int(disp_h * (infer_w / max(disp_w, 1))), 32)
                    infer_frame = cv2.resize(img_bgr, (infer_w, infer_h))

                    # 2. YOLO Prediction
                    results = yolo_model.predict(
                        source=infer_frame,
                        imgsz=640,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )

                    scale_x = disp_w / infer_w
                    scale_y = disp_h / infer_h

                    annotated_frame = img_bgr.copy()
                    detected_count = 0

                    for r in results:
                        for box in r.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf_val = float(box.conf[0].cpu().numpy())

                            # แปลงพิกัดกลับสู่ภาพต้นฉบับ
                            x1 = max(0, int(xyxy[0] * scale_x))
                            y1 = max(0, int(xyxy[1] * scale_y))
                            x2 = min(disp_w, int(xyxy[2] * scale_x))
                            y2 = min(disp_h, int(xyxy[3] * scale_y))

                            if (x2 - x1) < 10 or (y2 - y1) < 10:
                                continue

                            detected_count += 1
                            plate_crop = img_bgr[y1:y2, x1:x2].copy()

                            # Preprocessing สำหรับ OCR (Resize, Grayscale, CLAHE)
                            preprocessed_crop = preprocess_for_ocr(plate_crop)

                            # EasyOCR Reading
                            valid_words = []
                            conf_scores = []
                            try:
                                ocr_results = ocr_reader.readtext(preprocessed_crop)
                                for res in ocr_results:
                                    _, ocr_text, prob = res
                                    if prob >= ocr_conf_threshold:
                                        valid_words.append(ocr_text)
                                        conf_scores.append(prob)
                            except Exception:
                                valid_words = []
                                conf_scores = []

                            raw_plate_text = clean_plate_text(" ".join(valid_words))
                            avg_ocr_conf = float(sum(conf_scores) / len(conf_scores)) if conf_scores else 0.0

                            # ข้อความป้ายทะเบียน: ถ้าอ่านไม่ออกหรือเป็นค่าว่าง ให้ใส่ "ไม่ทราบป้าย"
                            final_text = raw_plate_text if raw_plate_text.strip() else "ไม่ทราบป้าย"

                            # -------------------------------------------------------------
                            # เงื่อนไขที่ 1 & 3: ข้ามระบบเช็คซ้ำ (Deduplication) สำหรับโหมดรูปภาพ
                            # บังคับบันทึกข้อมูลภาพที่ครอป (crop_img) ลงใน st.session_state.detected_records เสมอ
                            # แม้ว่า EasyOCR จะอ่านข้อความไม่ออกก็ตาม
                            # -------------------------------------------------------------
                            crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            new_record = {
                                'id': len(st.session_state.detected_records) + 1,
                                'timestamp': now_str,
                                'text': final_text,
                                'yolo_conf': float(conf_val),
                                'ocr_conf': float(avg_ocr_conf),
                                'image': crop_rgb
                            }
                            st.session_state.detected_records.append(new_record)

                            # วาดกรอบสไตล์ Cyber CCTV
                            label_to_draw = final_text if final_text != "ไม่ทราบป้าย" else "PLATE"
                            annotated_frame = draw_cctv_bounding_box(
                                frame=annotated_frame,
                                x1=x1, y1=y1, x2=x2, y2=y2,
                                label=label_to_draw,
                                conf=conf_val
                            )

                    proc_time = time.time() - t_start
                    simulated_fps = 1.0 / max(proc_time, 0.001)

                    # วาด OSD Overlay
                    annotated_frame = draw_osd_overlay(annotated_frame, simulated_fps)

                    # บันทึกผลลัพธ์ภาพที่ตรวจจับลงใน session_state เพื่อคงการแสดงผลไว้หลัง st.rerun()
                    st.session_state.image_annotated_result = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.image_detection_summary = {
                        "count": detected_count,
                        "time": proc_time
                    }

                    # เงื่อนไขที่ 2: รันคำสั่ง st.rerun() ทันที เพื่อบังคับให้แผงประวัติด้านขวาอัปเดตข้อมูลใหม่ล่าสุด
                    st.rerun()

            # แสดงผลภาพในโหมดรูปภาพ
            if st.session_state.get("image_annotated_result") is not None:
                st.image(
                    st.session_state.image_annotated_result,
                    caption="ผลลัพธ์การตรวจจับป้ายทะเบียน",
                    channels="RGB",
                    use_container_width=True
                )
                summary = st.session_state.get("image_detection_summary")
                if summary:
                    if summary["count"] > 0:
                        st.success(f"🎯 ตรวจพบป้ายทะเบียนทั้งหมด {summary['count']} ป้าย ในเวลา {summary['time']:.2f} วินาที")
                    else:
                        st.warning("⚠️ ไม่พบป้ายทะเบียนในภาพที่มีความมั่นใจผ่านเกณฑ์")
            else:
                # แสดงภาพตัวอย่างต้นฉบับก่อนเริ่มตรวจจับ
                st.image(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                    caption="ภาพต้นฉบับที่อัปโหลด",
                    channels="RGB",
                    use_container_width=True
                )
    else:
        # หากไม่มีการอัปโหลดไฟล์ ให้รีเซ็ตแคชภาพของโหมดรูปภาพ
        st.session_state.last_uploaded_img_name = None
        st.session_state.image_annotated_result = None
        st.session_state.image_detection_summary = None


# ==============================================================================
# วาดแถบประวัติด้านขวาตอนโหลดหน้าเว็บ
# ==============================================================================
# หากไม่อยู่ระหว่างการสตรีมวิดีโอ ให้เรนเดอร์ประวัติตามปกติ
if not st.session_state.video_running:
    render_detection_history(col_results)
