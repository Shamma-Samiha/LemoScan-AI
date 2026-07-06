"""Polished Streamlit interface for LemoScan AI."""

import base64
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import streamlit as st

from utils.model_loader import get_model
from utils.prediction import predict_disease
from utils.xai import generate_gradcam


st.set_page_config(page_title="LemoScan AI", layout="wide")


# Keep the trained model in memory across Streamlit script reruns.
@st.cache_resource(show_spinner=False)
def load_model_once():
    """Return the shared model used by prediction and Grad-CAM."""
    return get_model()


def image_to_data_uri(image):
    """Convert a PIL image to an embeddable JPEG data URI."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def show_fitted_image(image, alt_text, frame_class):
    """Display a controlled-size image inside a styled frame."""
    safe_alt_text = escape(alt_text)
    image_uri = image_to_data_uri(image)
    st.markdown(
        f"""
        <div class="image-frame {frame_class}">
            <img src="{image_uri}" alt="{safe_alt_text}">
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_analysis():
    """Clear the uploaded file and all saved analysis results."""
    st.session_state.analysis_result = None
    st.session_state.analysis_file_id = None
    st.session_state.uploader_version += 1


# Initialize the small pieces of state used by the interactive workflow.
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_file_id" not in st.session_state:
    st.session_state.analysis_file_id = None
if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


# Custom CSS creates the green, cream, and white agricultural AI theme.
st.markdown(
    """
    <style>
        :root {
            color-scheme: light;
        }

        #MainMenu,
        footer,
        header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        .stDeployButton {
            display: none !important;
        }

        html, body, .stApp,
        [data-testid="stAppViewContainer"] {
            color: #18351f;
            background: #e8efe2;
        }

        [data-testid="stAppViewContainer"] {
            min-height: 100vh;
            background:
                radial-gradient(circle at 7% 5%, rgba(78, 126, 62, 0.22), transparent 26rem),
                radial-gradient(circle at 94% 92%, rgba(202, 164, 61, 0.16), transparent 24rem),
                linear-gradient(145deg, #f1edda 0%, #e1edda 100%);
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            width: calc(100% - 2rem);
            max-width: 1050px;
            margin: 1.35rem auto;
            padding: 2.1rem 2.6rem 2.4rem;
            background: rgba(255, 254, 248, 0.98);
            border: 1px solid rgba(55, 96, 48, 0.16);
            border-radius: 26px;
            box-shadow: 0 22px 62px rgba(31, 68, 35, 0.16);
        }

        p, label, small, span,
        [data-testid="stMarkdownContainer"] {
            color: #2f4633;
        }

        h1, h2, h3,
        [data-testid="stHeadingWithActionElements"] {
            color: #1f502b !important;
        }

        .hero {
            padding: 0 0.5rem 1.45rem;
            text-align: center;
        }

        .hero-kicker {
            margin: 0 0 0.45rem !important;
            color: #4e7d42 !important;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.17em;
            text-transform: uppercase;
        }

        .hero h1 {
            margin: 0;
            color: #184d27 !important;
            font-size: clamp(2.45rem, 7vw, 3.85rem);
            font-weight: 850;
            letter-spacing: -0.045em;
            line-height: 1;
        }

        .hero-subtitle {
            margin: 0.7rem 0 0 !important;
            color: #315f38 !important;
            font-size: 1.15rem;
            font-weight: 650;
        }

        .hero-description {
            max-width: 720px;
            margin: 0.7rem auto 1rem !important;
            color: #516657 !important;
            font-size: 0.96rem;
            line-height: 1.55;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 0.5rem;
        }

        .feature-badge {
            padding: 0.34rem 0.72rem;
            color: #285b31;
            background: #e8f1df;
            border: 1px solid #c5dbb7;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 750;
        }

        .section-heading {
            margin: 0 0 0.15rem !important;
            color: #214c29 !important;
            font-size: 1.2rem;
            font-weight: 800;
        }

        .section-support {
            margin: 0 0 0.8rem !important;
            color: #627066 !important;
            font-size: 0.88rem;
        }

        .upload-panel {
            padding: 1.15rem 1.25rem 0.7rem;
            background: #f7f5e9;
            border: 1px solid #d9e2cf;
            border-radius: 19px;
        }

        [data-testid="stFileUploader"] {
            margin-bottom: 0.2rem;
        }

        [data-testid="stFileUploader"] > label {
            display: none;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 126px;
            padding: 1rem !important;
            background: #f1f5e9 !important;
            border: 2px dashed #5f9650 !important;
            border-radius: 15px !important;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            background: #e9f2e1 !important;
            border-color: #35743b !important;
        }

        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {
            color: #314f35 !important;
        }

        [data-testid="stFileUploaderDropzone"] svg {
            color: #34733c !important;
            fill: #34733c !important;
        }

        [data-testid="stFileUploaderDropzone"] button {
            color: #ffffff !important;
            background: #2f733a !important;
            border: 1px solid #2f733a !important;
            font-weight: 750;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #245d2f !important;
            border-color: #245d2f !important;
        }

        .format-note {
            margin: 0.15rem 0 0.1rem !important;
            color: #68766a !important;
            font-size: 0.78rem;
            text-align: right;
        }

        .info-box {
            margin-top: 0.9rem;
            padding: 0.85rem 1rem;
            color: #2d5633;
            background: #e8f2e1;
            border: 1px solid #c5dcb8;
            border-left: 5px solid #4e8d45;
            border-radius: 12px;
            font-size: 0.9rem;
        }

        .preview-label {
            margin: 1rem 0 0.45rem !important;
            color: #315638 !important;
            font-size: 0.86rem;
            font-weight: 750;
        }

        .image-frame {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            overflow: hidden;
            background:
                linear-gradient(45deg, #eff2e9 25%, transparent 25%),
                linear-gradient(-45deg, #eff2e9 25%, transparent 25%),
                #f8f9f4;
            background-size: 18px 18px;
            border: 1px solid #d6e0d0;
            border-radius: 17px;
            box-shadow: 0 9px 25px rgba(34, 73, 39, 0.1);
        }

        .image-frame img {
            display: block;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .preview-frame {
            max-width: 330px;
            height: 210px;
            margin: 0 auto 0.9rem;
        }

        .result-frame {
            height: 350px;
        }

        .comparison-frame {
            height: 320px;
        }

        .result-card {
            padding: 1.05rem 1.2rem;
            margin-bottom: 0.85rem;
            background: linear-gradient(135deg, #ffffff, #f1f6ec);
            border: 1px solid #cfdfc5;
            border-left: 5px solid #44863f;
            border-radius: 15px;
            box-shadow: 0 8px 22px rgba(35, 75, 38, 0.08);
        }

        .result-label {
            margin: 0 0 0.28rem !important;
            color: #607062 !important;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.075em;
            text-transform: uppercase;
        }

        .result-value {
            margin: 0 !important;
            color: #1e542a !important;
            font-size: clamp(1.3rem, 4vw, 1.75rem);
            font-weight: 850;
        }

        .result-note {
            padding: 0.85rem 0.95rem;
            color: #536258;
            background: #f4f4e8;
            border-radius: 11px;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .section-divider {
            height: 1px;
            margin: 1.75rem 0 1.4rem;
            background: linear-gradient(90deg, transparent, #c4d5bc, transparent);
        }

        .comparison-title {
            margin: 0.45rem 0 0.45rem !important;
            color: #355b3a !important;
            font-size: 0.86rem;
            font-weight: 800;
            text-align: center;
        }

        .attention-note {
            margin: 0.8rem 0 0 !important;
            color: #697168 !important;
            font-size: 0.8rem;
            text-align: center;
        }

        .disclaimer {
            margin-top: 1.25rem;
            padding: 0.95rem 1.05rem;
            color: #554819;
            background: #fbf3d5;
            border: 1px solid #e5d397;
            border-left: 5px solid #a98727;
            border-radius: 12px;
            line-height: 1.5;
        }

        .app-footer {
            margin: 1.4rem 0 -0.7rem !important;
            color: #738074 !important;
            font-size: 0.75rem;
            text-align: center;
        }

        [data-testid="stAlert"] {
            border-radius: 12px !important;
        }

        [data-testid="stAlert"][data-baseweb="notification"],
        div[data-testid="stAlert"] {
            background: #fff3c4 !important;
            border: 1px solid #dfbd46 !important;
        }

        [data-testid="stAlert"] p {
            color: #5e4b0e !important;
        }

        .stButton button {
            min-height: 2.75rem;
            border-radius: 11px;
            font-weight: 750;
        }

        .stButton button[kind="primary"] {
            color: #ffffff !important;
            background: #287038 !important;
            border-color: #287038 !important;
            box-shadow: 0 7px 17px rgba(40, 112, 56, 0.2);
        }

        .stButton button[kind="primary"]:hover {
            background: #1f5b2e !important;
            border-color: #1f5b2e !important;
        }

        .stButton button[kind="secondary"] {
            color: #315338 !important;
            background: #f8faf5 !important;
            border-color: #a9c29f !important;
        }

        [data-testid="stToggle"] label,
        [data-testid="stToggle"] span {
            color: #315338 !important;
        }

        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"],
            .block-container {
                width: calc(100% - 0.8rem);
                margin: 0.4rem auto;
                padding: 1.35rem 1rem 1.8rem;
                border-radius: 18px;
            }

            .hero {
                padding: 0 0 1.1rem;
            }

            .hero-subtitle {
                font-size: 1rem;
            }

            .hero-description {
                font-size: 0.88rem;
            }

            .upload-panel {
                padding: 0.9rem 0.9rem 0.5rem;
            }

            [data-testid="stFileUploaderDropzone"] {
                min-height: 112px;
                padding: 0.75rem !important;
            }

            .result-frame {
                height: 300px;
            }

            .comparison-frame {
                height: 275px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Hero content and model feature badges.
st.markdown(
    """
    <section class="hero">
        <p class="hero-kicker">Explainable plant health screening</p>
        <h1>LemoScan AI</h1>
        <p class="hero-subtitle">An Explainable AI System for Lemon Leaf Disease Detection</p>
        <p class="hero-description">
            Upload a lemon leaf image to detect possible disease conditions and
            view AI explanation using Grad-CAM.
        </p>
        <div class="badge-row">
            <span class="feature-badge">ResNet50</span>
            <span class="feature-badge">Grad-CAM</span>
            <span class="feature-badge">6 Classes</span>
            <span class="feature-badge">AI Assisted</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


# Upload area. Its key changes when the user clears the current analysis.
st.markdown(
    """
    <div class="upload-panel">
        <p class="section-heading">Upload a leaf image</p>
        <p class="section-support">Choose a clear, well-lit photo with the leaf in focus.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a lemon leaf image",
    type=["png", "jpg", "jpeg"],
    help="Choose a clear image of one lemon leaf.",
    key=f"leaf_uploader_{st.session_state.uploader_version}",
)

st.markdown(
    '<p class="format-note">Supported formats: PNG, JPG, JPEG</p>',
    unsafe_allow_html=True,
)


if uploaded_file is None:
    # Keep the main action visible so the workflow is obvious before upload.
    analyze_without_upload = st.button(
        "Analyze Leaf",
        type="primary",
        use_container_width=True,
        help="Upload a leaf image, then select Analyze Leaf.",
    )
    if analyze_without_upload:
        st.warning("Please upload a lemon leaf image before starting analysis.")
    st.markdown(
        '<div class="info-box">Upload a lemon leaf image to begin the analysis.</div>',
        unsafe_allow_html=True,
    )
else:
    uploaded_bytes = uploaded_file.getvalue()
    current_file_id = sha256(uploaded_bytes).hexdigest()

    # A newly selected file should never display results from the previous image.
    if st.session_state.analysis_file_id != current_file_id:
        st.session_state.analysis_result = None

    try:
        with Image.open(BytesIO(uploaded_bytes)) as image:
            uploaded_preview = image.convert("RGB").copy()
    except OSError:
        uploaded_preview = None
        st.error("The uploaded file is not a valid image. Please choose another file.")

    current_result = st.session_state.analysis_result

    if uploaded_preview is not None:
        # Show a compact preview before analysis, not another oversized image.
        if current_result is None:
            st.markdown(
                '<p class="preview-label">Ready to analyze</p>',
                unsafe_allow_html=True,
            )
            show_fitted_image(uploaded_preview, "Uploaded lemon leaf preview", "preview-frame")

        analyze_column, clear_column = st.columns(2, gap="small")
        with analyze_column:
            analyze_clicked = st.button(
                "Analyze Leaf",
                type="primary",
                use_container_width=True,
            )
        with clear_column:
            st.button(
                "Clear / Upload Another Image",
                use_container_width=True,
                on_click=clear_analysis,
            )

        # Prediction runs only after this explicit button click.
        if analyze_clicked:
            try:
                with TemporaryDirectory(prefix="lemoscan_") as temporary_folder:
                    temporary_folder = Path(temporary_folder)
                    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
                    image_path = temporary_folder / f"uploaded_leaf{suffix}"
                    gradcam_path = temporary_folder / "gradcam_result.jpg"
                    image_path.write_bytes(uploaded_bytes)

                    with st.spinner("Analyzing leaf image..."):
                        load_model_once()
                        predicted_class, confidence, _, warning = predict_disease(
                            str(image_path)
                        )
                        generate_gradcam(str(image_path), str(gradcam_path))

                        with Image.open(gradcam_path) as image:
                            gradcam_preview = image.convert("RGB").copy()

                # Save result images in memory so UI toggles do not rerun the model.
                st.session_state.analysis_result = {
                    "uploaded_image": uploaded_preview,
                    "gradcam_image": gradcam_preview,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "warning": warning,
                }
                st.session_state.analysis_file_id = current_file_id
                st.rerun()

            except (OSError, ValueError, RuntimeError):
                st.session_state.analysis_result = None
                st.error("Analysis failed. Please try a clear image or upload another file.")

    # Render the saved result without running prediction again.
    result = st.session_state.analysis_result
    if result is not None and st.session_state.analysis_file_id == current_file_id:
        st.success("Analysis completed successfully.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        image_column, result_column = st.columns([1.08, 0.92], gap="large")

        with image_column:
            st.subheader("Uploaded Leaf Image")
            show_fitted_image(
                result["uploaded_image"],
                "Uploaded lemon leaf",
                "result-frame",
            )

        with result_column:
            st.subheader("Prediction Result")
            safe_predicted_class = escape(str(result["predicted_class"]))
            st.markdown(
                f"""
                <div class="result-card">
                    <p class="result-label">Predicted Disease</p>
                    <p class="result-value">{safe_predicted_class}</p>
                </div>
                <div class="result-card">
                    <p class="result-label">Confidence Score</p>
                    <p class="result-value">{result["confidence"]:.2f}%</p>
                </div>
                <div class="result-note">
                    The model compares visual leaf patterns learned during training.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if result["warning"]:
                st.warning(result["warning"])

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("Grad-CAM Explanation")
        st.markdown(
            '<p class="section-support">Highlighted regions show the areas that '
            'influenced the model prediction.</p>',
            unsafe_allow_html=True,
        )

        show_gradcam = st.toggle(
            "Show Grad-CAM comparison",
            value=True,
            key=f"show_gradcam_{current_file_id}",
        )

        if show_gradcam:
            original_column, gradcam_column = st.columns(2, gap="medium")
            with original_column:
                st.markdown(
                    '<p class="comparison-title">Original Image</p>',
                    unsafe_allow_html=True,
                )
                show_fitted_image(
                    result["uploaded_image"],
                    "Original lemon leaf",
                    "comparison-frame",
                )

            with gradcam_column:
                st.markdown(
                    '<p class="comparison-title">Model Attention</p>',
                    unsafe_allow_html=True,
                )
                show_fitted_image(
                    result["gradcam_image"],
                    "Grad-CAM model attention",
                    "comparison-frame",
                )

            st.markdown(
                '<p class="attention-note">Red/yellow regions indicate stronger '
                'model attention.</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="disclaimer"><strong>Important:</strong> This result is '
            'AI-assisted and should not replace expert agricultural advice.</div>',
            unsafe_allow_html=True,
        )


st.markdown(
    '<p class="app-footer">LemoScan AI &middot; Explainable lemon leaf screening</p>',
    unsafe_allow_html=True,
)
