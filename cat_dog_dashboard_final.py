import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cat_dog_model.tflite")


# =========================================================
# LOAD TFLITE MODEL
# =========================================================

@st.cache_resource
def load_tflite_model():

    if not os.path.exists(MODEL_PATH):
        return None, None

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


model_data = load_tflite_model()

if model_data[0] is None:
    interpreter = None
    input_details = None
    output_details = None
else:
    interpreter, input_details, output_details = model_data


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #080d1c;
        color: white;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9aa6bd;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .result-card {
        background: #111827;
        border: 1px solid #26324a;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
    }

    .prediction {
        font-size: 38px;
        font-weight: 800;
        margin: 15px 0;
    }

    .confidence {
        font-size: 24px;
        font-weight: 700;
    }

    .small-text {
        color: #9aa6bd;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🐾 Cat vs Dog AI")

    st.markdown("---")

    st.markdown("### ⚙️ Model Information")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Input Size:** 128 × 128")
    st.write("**Classes:** Cat / Dog")
    st.write("**Output:** Sigmoid")
    st.write("**Task:** Binary Classification")

    st.markdown("---")

    st.markdown("### 🤖 Model Status")

    if interpreter is not None:
        st.success("🟢 Model Online")
        st.caption("TFLite model loaded successfully")
    else:
        st.error("🔴 Model Not Found")
        st.caption("cat_dog_model.tflite is missing")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🐾 Cat vs Dog AI Classifier</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Deep Learning Image Classification using MobileNetV2</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TOP METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🐱🐶 Classes", "2")

with col2:
    st.metric("🖼️ Input Size", "128 × 128")

with col3:
    st.metric("🧠 AI Model", "MobileNetV2")

with col4:
    if interpreter is not None:
        st.metric("📡 Status", "ONLINE")
    else:
        st.metric("📡 Status", "OFFLINE")


st.markdown("---")


# =========================================================
# UPLOAD + PREVIEW
# =========================================================

left, right = st.columns(2)

with left:

    st.markdown("## 📤 Upload Image")

    st.markdown(
        """
        <div class="card">
            <h3>Choose a Cat or Dog Image</h3>
            <p class="small-text">
            Upload a JPG, JPEG, or PNG image.
            The trained deep-learning model will classify it.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )


with right:

    st.markdown("## 🖼️ Image Preview")

    if uploaded_file is None:

        st.markdown(
            """
            <div class="result-card" style="height:300px;">
                <div style="font-size:70px;">🐶 🐱</div>
                <h2>No Image Selected</h2>
                <p class="small-text">
                Upload an image to begin AI classification.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True,
        )


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.markdown("---")

    if interpreter is None:

        st.error(
            "Model not found. Make sure cat_dog_model.tflite "
            "is uploaded to the GitHub repository."
        )

    else:

        image = Image.open(uploaded_file).convert("RGB")

        # ---------------------------------------------
        # Resize according to actual TFLite model
        # ---------------------------------------------

        input_shape = input_details[0]["shape"]

        height = int(input_shape[1])
        width = int(input_shape[2])

        resized_image = image.resize((width, height))

        img_array = np.array(
            resized_image,
            dtype=np.float32
        )

        img_array = np.expand_dims(img_array, axis=0)

        # ---------------------------------------------
        # Normalize
        # ---------------------------------------------

        img_array = img_array / 255.0

        # ---------------------------------------------
        # Handle input dtype
        # ---------------------------------------------

        input_dtype = input_details[0]["dtype"]

        if input_dtype != np.float32:

            scale, zero_point = input_details[0]["quantization"]

            if scale != 0:
                img_array = img_array / scale + zero_point

            img_array = img_array.astype(input_dtype)

        else:

            img_array = img_array.astype(np.float32)

        # ---------------------------------------------
        # Run TFLite prediction
        # ---------------------------------------------

        interpreter.set_tensor(
            input_details[0]["index"],
            img_array
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        prediction = float(output[0][0])

        # ---------------------------------------------
        # Cat / Dog classification
        # ---------------------------------------------

        if prediction >= 0.5:

            result = "DOG 🐶"
            confidence = prediction * 100

            dog_probability = prediction * 100
            cat_probability = (1 - prediction) * 100

        else:

            result = "CAT 🐱"
            confidence = (1 - prediction) * 100

            cat_probability = (1 - prediction) * 100
            dog_probability = prediction * 100


        # =================================================
        # RESULT
        # =================================================

        st.markdown("## 🔮 Prediction")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Prediction",
                result
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with result_col3:

            st.metric(
                "Model",
                "MobileNetV2"
            )


        st.markdown("### 📊 Class Probabilities")

        prob1, prob2 = st.columns(2)

        with prob1:

            st.write(f"🐱 **Cat:** {cat_probability:.2f}%")

            st.progress(
                int(cat_probability)
            )

        with prob2:

            st.write(f"🐶 **Dog:** {dog_probability:.2f}%")

            st.progress(
                int(dog_probability)
            )


        # =================================================
        # FINAL RESULT
        # =================================================

        st.markdown(
            f"""
            <div class="result-card">
                <div class="small-text">AI CLASSIFICATION RESULT</div>
                <div class="prediction">{result}</div>
                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#7f8ba3;">
        🐾 Cat vs Dog Image Classification |
        TensorFlow Lite + MobileNetV2 + Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
)
