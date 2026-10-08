import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.keras")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 35px;
        padding-bottom: 30px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 46px;
        font-weight: 800;
        margin: 0;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 18px;
        margin-top: 8px;
        margin-bottom: 28px;
    }

    .box {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,.22);
    }

    .box-title {
        color: #f8fafc;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .box-text {
        color: #94a3b8;
        font-size: 15px;
    }

    .empty-preview {
        min-height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        background: #111827;
        border: 1px dashed #334155;
        border-radius: 18px;
        padding: 30px;
    }

    .empty-icon {
        font-size: 58px;
        margin-bottom: 12px;
    }

    .empty-title {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
    }

    .empty-text {
        color: #94a3b8;
        font-size: 15px;
    }

    .result-box {
        background: linear-gradient(135deg,#172033,#111827);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,.28);
    }

    .result-label {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .result-value {
        color: #f8fafc;
        font-size: 42px;
        font-weight: 800;
        margin-top: 12px;
    }

    .result-confidence {
        color: #38bdf8;
        font-size: 30px;
        font-weight: 700;
        margin-top: 8px;
    }

    .result-text {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 5px;
    }

    .footer {
        color: #64748b;
        text-align: center;
        padding: 22px;
    }

    section[data-testid="stSidebar"] {
        background: #080d19;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🐾 Cat vs Dog AI")
    st.caption("Deep Learning Image Classification")

    st.divider()

    st.markdown("### 🧠 Technology")
    st.write("🐍 Python")
    st.write("🧠 TensorFlow")
    st.write("🔗 Keras")
    st.write("📊 Streamlit")

    st.divider()

    st.markdown("### ⚙️ Model Information")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Input Size:** 160 × 160")
    st.write("**Classes:** Cat / Dog")
    st.write("**Output:** Sigmoid")
    st.write("**Task:** Binary Classification")

    st.divider()

    st.markdown("### 📡 Model Status")
    if model is not None:
        st.success("🟢 Model Loaded")
    else:
        st.error("🔴 Model Not Found")
        st.caption("Put cat_dog_model.keras beside this Python file.")

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">🐾 Cat vs Dog AI Image Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">Deep Learning powered image classification dashboard</div>',
    unsafe_allow_html=True,
)

# ============================================================
# TOP STATS - NATIVE STREAMLIT, NO HTML
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("🐶 🐱 Classes", "2", "Cat & Dog")

with c2:
    st.metric("🖼️ Input Size", "160 × 160", "RGB Image")

with c3:
    st.metric("🧠 AI Model", "MobileNetV2", "Transfer Learning")

with c4:
    st.metric("📡 Status", "ONLINE" if model is not None else "OFFLINE",
              "Model Loaded" if model is not None else "Model Missing")

st.write("")
st.divider()
st.write("")

# ============================================================
# UPLOAD + PREVIEW
# ============================================================

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("📤 Upload Image")

    st.markdown(
        '<div class="box">'
        '<div class="box-title">Choose a Cat or Dog Image</div>'
        '<div class="box-text">Upload a JPG, JPEG, or PNG image. '
        'The trained deep-learning model will classify it.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )

with right:
    st.subheader("🖼️ Image Preview")

    if uploaded_file is None:
        st.markdown(
            '<div class="empty-preview">'
            '<div class="empty-icon">🐶 🐱</div>'
            '<div class="empty-title">No Image Selected</div>'
            '<div class="empty-text">Upload an image to begin AI classification.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:
    st.write("")
    st.divider()
    st.subheader("🤖 AI Prediction")

    if model is None:
        st.error("Model not found.")
        st.warning(
            "Make sure cat_dog_model.keras is in the same folder as this Python file."
        )
    else:
        resized = image.resize((160, 160))
        img_array = np.array(resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = float(model.predict(img_array, verbose=0)[0][0])

        dog_probability = prediction
        cat_probability = 1.0 - prediction

        if prediction >= 0.5:
            result = "DOG 🐶"
            confidence = dog_probability * 100
        else:
            result = "CAT 🐱"
            confidence = cat_probability * 100

        r1, r2 = st.columns([1, 1], gap="large")

        with r1:
            st.markdown(
                f'<div class="result-box">'
                f'<div class="result-label">AI Classification</div>'
                f'<div class="result-value">{result}</div>'
                f'<div class="result-confidence">{confidence:.2f}%</div>'
                f'<div class="result-text">Prediction Confidence</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                '<div class="box">'
                '<div class="box-title">📊 Class Probabilities</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            st.write(f"🐱 Cat — **{cat_probability * 100:.2f}%**")
            st.progress(cat_probability)

            st.write(f"🐶 Dog — **{dog_probability * 100:.2f}%**")
            st.progress(dog_probability)

        st.write("")
        st.subheader("🔬 Model Information")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Architecture", "MobileNetV2")

        with m2:
            st.metric("Input Size", "160 × 160")

        with m3:
            st.metric("Classes", "2")

        with m4:
            st.metric("Activation", "Sigmoid")

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    '🐾 Cat vs Dog Image Classification • '
    'TensorFlow • Keras • MobileNetV2 • Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
