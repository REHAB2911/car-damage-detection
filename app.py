import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import datetime

from cost_estimator import get_cost_estimate, format_mad, get_confidence_badge
from pdf_report import generate_report
from translations import t
from explainability import get_explanation

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="InsurTech — Détection de dommages", layout="wide", page_icon="🚗")
MODEL_PATH = r"C:\projets\car_damage_cnn\best_model_final2.h5"
CLASS_NAMES = ['leger', 'moyen', 'severe']
COLORS = {'leger': '🟢', 'moyen': '🟡', 'severe': '🔴'}
THRESHOLDS = {'severe': 0.50, 'moyen': 0.35}

SEVERITY_HEX = {
    'leger':  ('#0d1a0d', '#22c55e', '#f0fdf4'),
    'moyen':  ('#1a1200', '#f59e0b', '#fffbeb'),
    'severe': ('#1a0000', '#ef4444', '#fef2f2'),
}

# ── Initialize session state for analytics ────────────────────────────────────
if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
if 'total_images_processed' not in st.session_state:
    st.session_state.total_images_processed = 0

# ── Premium CSS with Light Theme + Glassmorphism + Navy ───────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    color: #1a2a3a;
}

/* ── Glassmorphism effect ── */
.glass-card {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
}

/* ── Sidebar with glass effect ── */
section[data-testid="stSidebar"] {
    background: rgba(10, 35, 66, 0.92);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255, 255, 255, 0.15);
}
section[data-testid="stSidebar"] * { 
    color: rgba(255, 255, 255, 0.85) !important; 
}
section[data-testid="stSidebar"] hr { 
    border-color: rgba(255, 255, 255, 0.1) !important; 
}
section[data-testid="stSidebar"] .stRadio label {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #e2e8f0; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Top wordmark bar ── */
.wordmark-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 20px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 40px;
}
.wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0a2342;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.wordmark span {
    color: #1e3a5f;
    font-weight: 700;
}
.wordmark-badge {
    background: rgba(30, 58, 95, 0.08);
    border: 1px solid rgba(30, 58, 95, 0.15);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.7rem;
    color: #1e3a5f;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* ── Hero headline ── */
.hero {
    padding: 40px 0 48px 0;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #1e3a5f;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #0a2342;
    line-height: 1.15;
    margin: 0 0 16px 0;
}
.hero-title em {
    color: #1e3a5f;
    font-style: italic;
}
.hero-sub {
    font-size: 0.95rem;
    color: #5a6e7a;
    font-weight: 300;
    max-width: 520px;
    line-height: 1.7;
}

/* ── Section label ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 48px 0 20px 0;
}
.section-label-line {
    flex: 1;
    height: 1px;
    background: rgba(0, 0, 0, 0.08);
}
.section-label-text {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #1e3a5f;
    white-space: nowrap;
}
.section-label-num {
    font-family: 'Playfair Display', serif;
    font-size: 0.85rem;
    color: #1e3a5f;
    margin-right: 4px;
}

/* ── Photo strip card ── */
.photo-card {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 10px 12px 12px 12px;
    margin-top: 10px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}
.photo-card-severity {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.photo-card-probs {
    font-size: 0.7rem;
    color: #5a6e7a;
    margin-top: 4px;
    line-height: 1.8;
}

/* ── IMAGE FIX: contain instead of cover ── */
.stImage img,
[data-testid="stImage"] img {
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
    max-height: 320px !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.5) !important;
}

/* ── Verdict block ── */
.verdict-block {
    border-left: 3px solid #1e3a5f;
    padding: 28px 32px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border-radius: 0 12px 12px 0;
    margin-bottom: 4px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}
.verdict-eyebrow {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a6e7a;
    margin-bottom: 10px;
}
.verdict-main {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 8px;
}
.verdict-sub {
    font-size: 0.82rem;
    color: #5a6e7a;
    font-weight: 300;
}

/* ── Risk panel ── */
.risk-panel {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
}
.risk-panel-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a6e7a;
    margin-bottom: 16px;
}
.risk-panel-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.risk-panel-msg {
    font-size: 0.78rem;
    color: #5a6e7a;
    line-height: 1.6;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
    padding-top: 12px;
    margin-top: 4px;
}
.risk-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Prob bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.prob-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a6e7a;
    width: 60px;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    height: 3px;
    background: rgba(0, 0, 0, 0.08);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.prob-pct {
    font-size: 0.75rem;
    color: #1e3a5f;
    font-weight: 500;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Grad-CAM labels ── */
.img-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5a6e7a;
    text-align: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
}

/* ── Explanation ── */
.explanation {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 20px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}
.explanation-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
    color: #1e3a5f;
}
.explanation-label {
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1e3a5f;
    margin-bottom: 6px;
}
.explanation-text {
    font-size: 0.88rem;
    color: #5a6e7a;
    line-height: 1.7;
    font-weight: 300;
}

/* ── Cost grid ── */
.cost-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 16px;
}
.cost-cell {
    background: rgba(255, 255, 255, 0.85);
    padding: 22px 24px;
    text-align: center;
}
.cost-cell.featured { 
    background: rgba(30, 58, 95, 0.08);
}
.cost-label {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5a6e7a;
    margin-bottom: 10px;
}
.cost-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0a2342;
}
.cost-value.gold { color: #1e3a5f; }

/* ── Info strip ── */
.info-strip {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.82rem;
    color: #5a6e7a;
    line-height: 1.7;
}
.info-strip strong { color: #1e3a5f; font-weight: 500; }

/* ── Rec list ── */
.rec-item {
    padding: 11px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    font-size: 0.83rem;
    color: #5a6e7a;
    display: flex;
    gap: 12px;
    align-items: flex-start;
    line-height: 1.5;
}
.rec-item:last-child { border-bottom: none; }
.rec-bullet {
    color: #1e3a5f;
    font-size: 0.6rem;
    margin-top: 5px;
    flex-shrink: 0;
}

/* ── PDF section ── */
.pdf-row {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    margin-bottom: 16px;
}
.pdf-icon { font-size: 1.4rem; color: #1e3a5f; }
.pdf-label { font-size: 0.82rem; color: #5a6e7a; }
.pdf-title { font-size: 0.9rem; color: #0a2342; font-weight: 500; }

/* ── Streamlit overrides ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1e3a5f, #2c5282) !important;
    border-radius: 2px !important;
}
.stProgress > div {
    background: rgba(0, 0, 0, 0.08) !important;
    border-radius: 2px !important;
    height: 3px !important;
}
.stButton > button {
    background: #1e3a5f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { 
    background: #2c5282 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(30, 58, 95, 0.2);
}
.stDownloadButton > button {
    background: transparent !important;
    color: #1e3a5f !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
}
.stDownloadButton > button:hover {
    background: #1e3a5f !important;
    color: #ffffff !important;
}
.stFileUploader {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(8px);
    border: 1px dashed rgba(30, 58, 95, 0.3) !important;
    border-radius: 12px !important;
}
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(30, 58, 95, 0.15) !important;
    border-radius: 8px !important;
    color: #0a2342 !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1e3a5f !important;
    box-shadow: 0 0 0 2px rgba(30, 58, 95, 0.1) !important;
}
.streamlit-expanderHeader {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(30, 58, 95, 0.1) !important;
    border-radius: 8px !important;
    color: #1e3a5f !important;
    font-size: 0.82rem !important;
}
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.7) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(30, 58, 95, 0.1) !important;
    border-radius: 8px !important;
}
.stAlert {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(8px);
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    border-left: 3px solid #1e3a5f !important;
}
hr { border-color: rgba(0, 0, 0, 0.08) !important; }
label { color: #5a6e7a !important; font-size: 0.78rem !important; letter-spacing: 0.06em !important; }

/* ── Custom glass container for images ── */
.image-container {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
}

/* ── Confidence badge styling ── */
.confidence-high {
    background: linear-gradient(135deg, #22c55e20, #22c55e10);
    border-left: 3px solid #22c55e;
}
.confidence-moderate {
    background: linear-gradient(135deg, #f59e0b20, #f59e0b10);
    border-left: 3px solid #f59e0b;
}
.confidence-low {
    background: linear-gradient(135deg, #ef444420, #ef444410);
    border-left: 3px solid #ef4444;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar with Logo/Thumbnail ───────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; margin-bottom:20px; padding: 20px 0 10px 0;">
    <div style="font-size:3.5rem; animation: pulse 2s infinite;">🚗💥</div>
    <div style="font-weight:700; color:#ffffff; font-size:1.2rem; letter-spacing:0.08em; margin-top:8px;">
        InsurTech<span style="color:#1e3a5f;"> AI</span>
    </div>
    <div style="font-size:0.65rem; color:rgba(255,255,255,0.6); letter-spacing:0.15em; margin-top:4px;">
        CAR DAMAGE ASSESSMENT
    </div>
</div>
<style>
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
</style>
""", unsafe_allow_html=True)

# ── Language toggle ───────────────────────────────────────────────────────────
lang = st.sidebar.radio("🌐 Langue / اللغة", ['fr', 'ar'], horizontal=True)
if lang == 'ar':
    st.markdown('<style>.stApp { direction: rtl; text-align: right; }</style>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:0.72rem; line-height:2; letter-spacing:0.06em;'>
<div style='color:#1e3a5f; font-weight:600; text-transform:uppercase; letter-spacing:0.15em; margin-bottom:8px;'>Système</div>
<div>Modèle — ResNet50V2</div>
<div>Classes — 3 niveaux</div>
<div>Marché — Maroc 🇲🇦</div>
<div>Photos — 1 à 5</div>
</div>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Core functions ────────────────────────────────────────────────────────────
def preprocess(img):
    img = img.resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def apply_thresholds(probs):
    if probs[2] >= THRESHOLDS['severe']:
        return 2, probs[2]
    elif probs[1] >= THRESHOLDS['moyen']:
        return 1, probs[1]
    else:
        return 0, probs[0]

def make_gradcam(img_array, model):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("post_bn").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions[0].numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    img_cv = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 1 - alpha, heatmap_colored, alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

def aggregate_results(all_probs):
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx, confidence = apply_thresholds(avg_probs)
    individual_classes = [apply_thresholds(p)[0] for p in all_probs]
    is_inconsistent = len(set(individual_classes)) > 1
    is_suspicious = (pred_idx == 2 and confidence < 0.65)
    return avg_probs, pred_idx, confidence, is_inconsistent, is_suspicious

def fraud_risk_level(is_inconsistent, is_suspicious, confidence):
    score = sum([is_inconsistent, is_suspicious, confidence < 0.60])
    levels = ['low', 'moderate', 'high']
    return score, levels[min(score, 2)]

def section_label(num, text):
    st.markdown(f"""
    <div class="section-label">
        <div class="section-label-line"></div>
        <div class="section-label-text">
            <span class="section-label-num">{num}&nbsp;&nbsp;</span>{text}
        </div>
        <div class="section-label-line"></div>
    </div>
    """, unsafe_allow_html=True)

def display_confidence_badge(confidence, lang):
    """Display prominent confidence badge"""
    if confidence > 0.85:
        st.success(f"🎯 **{t('high_confidence', lang)}** - {t('high_confidence_msg', lang)}")
    elif confidence > 0.65:
        st.info(f"📊 **{t('moderate_confidence', lang)}** - {t('moderate_confidence_msg', lang)}")
    else:
        st.warning(f"⚠️ **{t('low_confidence', lang)}** - {t('low_confidence_msg', lang)}")

# ── Wordmark bar ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="wordmark-bar">
    <div class="wordmark">Insur<span>Tech</span> &nbsp;·&nbsp; AI Damage Assessment</div>
    <div class="wordmark-badge">v2.0 · ResNet50V2</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Évaluation automatique des sinistres</div>
    <div class="hero-title">Analyse <em>intelligente</em><br>des dommages véhicule</div>
    <div class="hero-sub">{t('app_subtitle', lang).replace('**', '')}</div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    t('upload_label', lang),
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

st.caption("200MB par fichier • JPG, JPEG, PNG")

if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning(t('max_photos_warning', lang))
        uploaded_files = uploaded_files[:5]

    model = load_model()

    # ── Per-photo ─────────────────────────────────────────────────────────────
    section_label("01", t('per_photo_title', lang).upper())

    all_probs     = []
    all_overlays  = []
    all_originals = []
    all_heatmaps  = []
    worst_heatmap_idx = 0
    worst_severity    = -1

    cols = st.columns(len(uploaded_files))

    for i, uploaded in enumerate(uploaded_files):
        img       = Image.open(uploaded).convert('RGB')
        img_array = preprocess(img)
        heatmap, probs = make_gradcam(img_array, model)
        overlay   = overlay_gradcam(img, heatmap)
        pred_idx, conf_i = apply_thresholds(probs)

        all_probs.append(probs)
        all_overlays.append(overlay)
        all_originals.append((img_array[0] * 255).astype(np.uint8))
        all_heatmaps.append(heatmap)

        if pred_idx > worst_severity:
            worst_severity    = pred_idx
            worst_heatmap_idx = i

        label = CLASS_NAMES[pred_idx]
        _, sev_color, _ = SEVERITY_HEX[label]

        with cols[i]:
            # Wrap image in a fixed-height container for consistent sizing
            img_b64 = None
            import base64, io
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format='PNG')
            img_b64 = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="image-container" style="
                width:100%;
                height:200px;
                background:rgba(255,255,255,0.4);
                border-radius:12px;
                overflow:hidden;
                display:flex;
                align-items:center;
                justify-content:center;
            ">
                <img src="data:image/png;base64,{img_b64}"
                     style="max-width:100%; max-height:200px; object-fit:contain; display:block;" />
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="photo-card">
                <div class="photo-card-severity" style="color:{sev_color};">
                    {COLORS[label]} {t(label, lang).upper()} &nbsp;·&nbsp; {conf_i*100:.1f}%
                </div>
                <div class="photo-card-probs">
                    {t('leger', lang)} &nbsp;{probs[0]*100:.0f}%<br>
                    {t('moyen', lang)} &nbsp;{probs[1]*100:.0f}%<br>
                    {t('severe', lang)} &nbsp;{probs[2]*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Consolidated with Loading Animation ───────────────────────────────────
    section_label("02", t('consolidated_title', lang).upper())

    # Update analytics
    st.session_state.analysis_count += 1
    st.session_state.total_images_processed += len(uploaded_files)

    # Custom loading animation for aggregation
    with st.spinner("🧠 AI analyzing damage patterns across all images..."):
        import time
        time.sleep(0.3)  # Small delay to show spinner (optional)
        avg_probs, pred_idx, confidence, is_inconsistent, is_suspicious = aggregate_results(all_probs)
    
    pred_label = CLASS_NAMES[pred_idx]
    _, sev_color, _ = SEVERITY_HEX[pred_label]

    # Display confidence badge prominently
    display_confidence_badge(confidence, lang)

    if confidence < 0.60:
        st.warning(t('uncertainty_banner', lang))

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="verdict-block">
            <div class="verdict-eyebrow">{t('consolidated_label', lang)}</div>
            <div class="verdict-main" style="color:{sev_color};">
                {t(pred_label, lang).upper()}
            </div>
            <div class="verdict-sub">
                {confidence*100:.1f}% confiance &nbsp;·&nbsp; {t('based_on', lang, len(uploaded_files))}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        prob_bar_colors = {
            'leger':  '#22c55e',
            'moyen':  '#f59e0b',
            'severe': '#ef4444',
        }
        for cls in CLASS_NAMES:
            val = avg_probs[CLASS_NAMES.index(cls)]
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label">{t(cls, lang)}</div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{val*100:.1f}%; background:{prob_bar_colors[cls]};"></div>
                </div>
                <div class="prob-pct">{val*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        score, level = fraud_risk_level(is_inconsistent, is_suspicious, confidence)
        risk_colors  = {'low': '#22c55e', 'moderate': '#f59e0b', 'high': '#ef4444'}
        rc = risk_colors[level]
        risk_label = t(f'risk_{level}', lang)
        risk_msg   = t(f'risk_msg_{level}', lang)

        st.markdown(f"""
        <div class="risk-panel">
            <div class="risk-panel-label">{t('risk_title', lang)}</div>
            <div class="risk-panel-value">
                <span class="risk-dot" style="background:{rc};"></span>
                <span style="color:{rc};">{risk_label}</span>
            </div>
            <div class="risk-panel-msg">{risk_msg}</div>
        </div>
        """, unsafe_allow_html=True)

        if is_inconsistent:
            classes_per_photo = [t(CLASS_NAMES[apply_thresholds(p)[0]], lang) for p in all_probs]
            st.warning(t('inconsistent_warning', lang, ' · '.join([f'P{i+1}={c}' for i, c in enumerate(classes_per_photo)])))
        if is_suspicious:
            st.warning(t('suspicious_warning', lang))

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    section_label("03", t('gradcam_title', lang).upper())

    import base64, io as _io
    c1, c2 = st.columns(2, gap="medium")

    def img_to_b64(arr_or_pil, is_pil=False):
        buf = _io.BytesIO()
        if is_pil:
            arr_or_pil.save(buf, format='PNG')
        else:
            Image.fromarray(arr_or_pil).save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()

    with c1:
        best_img = Image.open(uploaded_files[worst_heatmap_idx]).convert('RGB')
        b64 = img_to_b64(best_img, is_pil=True)
        st.markdown(f"""
        <div class="image-container" style="width:100%;height:300px;background:rgba(255,255,255,0.4);border-radius:12px;
                    display:flex;align-items:center;justify-content:center;overflow:hidden;">
            <img src="data:image/png;base64,{b64}"
                 style="max-width:100%;max-height:300px;object-fit:contain;display:block;" />
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="img-label">{t("original_image", lang)}</div>', unsafe_allow_html=True)

    with c2:
        b64_ov = img_to_b64(all_overlays[worst_heatmap_idx])
        st.markdown(f"""
        <div class="image-container" style="width:100%;height:300px;background:rgba(255,255,255,0.4);border-radius:12px;
                    display:flex;align-items:center;justify-content:center;overflow:hidden;">
            <img src="data:image/png;base64,{b64_ov}"
                 style="max-width:100%;max-height:300px;object-fit:contain;display:block;" />
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="img-label">{t("gradcam_map", lang)}</div>', unsafe_allow_html=True)

    explanation = get_explanation(all_heatmaps[worst_heatmap_idx], lang)
    st.markdown(f"""
    <div class="explanation">
        <div class="explanation-icon">◈</div>
        <div>
            <div class="explanation-label">{t('explanation_label', lang)}</div>
            <div class="explanation-text">{explanation}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cost ──────────────────────────────────────────────────────────────────
    section_label("04", t('cost_title', lang).upper())

    estimate      = get_cost_estimate(pred_label, confidence, lang)
    conf_label, _ = get_confidence_badge(confidence, lang)

    st.markdown(f"""
    <div class="cost-grid">
        <div class="cost-cell">
            <div class="cost-label">{t('cost_low', lang)}</div>
            <div class="cost-value">{format_mad(estimate.min_cost)}</div>
        </div>
        <div class="cost-cell featured">
            <div class="cost-label">{t('cost_avg', lang)}</div>
            <div class="cost-value gold">{format_mad(estimate.avg_cost)}</div>
        </div>
        <div class="cost-cell">
            <div class="cost-label">{t('cost_high', lang)}</div>
            <div class="cost-value">{format_mad(estimate.max_cost)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-strip">
        ⏱ &nbsp;<strong>{estimate.repair_time}</strong>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        {t('model_confidence', lang)} &nbsp;<strong>{confidence*100:.1f}% ({conf_label})</strong>
        <br><br>
        <span style="color:#5a6e7a;">{estimate.description}</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(t('recommendations_label', lang)):
        for rec in estimate.recommendations:
            st.markdown(f"""
            <div class="rec-item">
                <div class="rec-bullet">◆</div>
                <div>{rec}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── PDF ───────────────────────────────────────────────────────────────────
    section_label("05", t('pdf_title', lang).upper())

    st.markdown(f"""
    <div class="pdf-row">
        <div class="pdf-icon">◎</div>
        <div>
            <div class="pdf-title">{t('pdf_title', lang)}</div>
            <div class="pdf-label">Rapport complet · Images · Estimation · Recommandations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(t('vehicle_info_label', lang)):
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            marque = st.text_input(t('brand', lang))
            annee  = st.text_input(t('year', lang))
        with col_v2:
            modele = st.text_input(t('model_car', lang))
            immat  = st.text_input(t('plate', lang))

    vehicle_info = None
    if any([marque, modele, annee, immat]):
        vehicle_info = {k: v for k, v in {
            t('brand', lang):     marque,
            t('model_car', lang): modele,
            t('year', lang):      annee,
            t('plate', lang):     immat,
        }.items() if v}

    if st.button(t('generate_pdf', lang)):
        with st.spinner(t('generating', lang)):
            probs_dict = {
                "leger":  float(avg_probs[0]),
                "moyen":  float(avg_probs[1]),
                "severe": float(avg_probs[2]),
            }
            pdf_bytes = generate_report(
                original_img=all_originals[worst_heatmap_idx],
                gradcam_img=all_overlays[worst_heatmap_idx],
                severity_label=estimate.severity,
                probabilities=probs_dict,
                confidence=confidence,
                cost_estimate=estimate,
                vehicle_info=vehicle_info,
                lang=lang,
            )
        st.success(t('pdf_success', lang))
        st.download_button(
            label=t('download_pdf', lang),
            data=pdf_bytes,
            file_name=f"rapport_sinistre_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

# ── Sidebar Analytics (always visible) ────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="margin-top: 20px;">
    <div style="font-size:0.65rem; letter-spacing:0.15em; text-transform:uppercase; color:rgba(255,255,255,0.4); margin-bottom:12px;">
        📊 STATISTIQUES DE SESSION
    </div>
</div>
""", unsafe_allow_html=True)

# Display analytics in sidebar
col_analytics1, col_analytics2 = st.sidebar.columns(2)
with col_analytics1:
    st.metric("Sinistres", st.session_state.analysis_count, delta=None)
with col_analytics2:
    st.metric("Images", st.session_state.total_images_processed, delta=None)

# Add a reset button for analytics
if st.sidebar.button("Réinitialiser", use_container_width=True):
    st.session_state.analysis_count = 0
    st.session_state.total_images_processed = 0
    st.rerun()

st.sidebar.caption("💡 Astuce : Téléchargez 3-5 photos pour une meilleure précision")