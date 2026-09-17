"""Streamlit presentation/controller. All model inference stays in shared utils."""
import base64
import json
from hashlib import sha256
from html import escape
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parent
CLASSES = ["Algal leaf spot", "Black spot", "Citrus canker", "Citrus pest", "Greening", "Healthy leaf"]
st.set_page_config(page_title="LemoScan AI | Explainable leaf analysis",page_icon=":material/eco:",layout="wide")

CSS = """
<style>
.stApp { background:#f5f7f3; color:#18342b; }
[data-testid="stSidebar"] { background:#eaf0e8; border-right:1px solid #dce5d9; }
[data-testid="stSidebar"] h2 { font-size:1.05rem; }
.block-container { max-width:1180px; padding-top:2.6rem; padding-bottom:2rem; }
h1,h2,h3 { letter-spacing:-.025em; color:#18342b; }
h1 { font-size:clamp(2.2rem,5vw,3.8rem)!important; line-height:1.08!important; }
h2 { font-size:1.6rem!important; }
.eyebrow { font-size:.72rem; letter-spacing:.16em; font-weight:700; color:#55725e; margin-bottom:.75rem; }
.hero { border-bottom:1px solid #d5e1d2; padding-bottom:1.4rem; margin-bottom:.5rem; }
.hero p { color:#52665b; max-width:730px; font-size:1rem; line-height:1.65; }
.subtitle { font-size:1.28rem!important; color:#315943!important; margin-bottom:.5rem; }
.chips { display:flex; gap:8px; flex-wrap:wrap; padding-top:8px; }
.chip { border:1px solid #cfddca; border-radius:100px; padding:5px 12px; font-size:.74rem; background:#edf3e9; color:#31533f; }
.result-card { padding:1.5rem; border:1px solid #cbdcc7; border-left:5px solid #397444; border-radius:14px; background:#eef5e9; }
.result-card h2 { margin:0; font-size:clamp(1.6rem,4vw,2.3rem)!important; }
.result-card .score { font-size:2rem; font-weight:700; margin-top:.6rem; }
.result-card p { margin:0; color:#46634e; }
.preview { border-radius:12px; overflow:hidden; border:1px solid #dce5d9; background:#edf1eb; }
.preview img { width:100%; height:330px; object-fit:contain; display:block; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }
[data-testid="stFileUploaderDropzone"] { background:#ffffff; border:1px dashed #9db29c; border-radius:12px; }
.stButton button { border-radius:9px; min-height:2.8rem; }
.stButton button[kind="primary"] { background:#245b3c; border-color:#245b3c; color:white; }
.stButton button[kind="primary"]:hover { background:#1b472e; border-color:#1b472e; color:white; }
[data-testid="stProgressBar"] > div > div > div > div { background:#4f8057; }
.sidebar-metric { background:#f5f7f3; border:1px solid #d4dfd0; border-radius:10px; padding:12px 14px; margin-bottom:8px; }
.sidebar-metric span { display:block; font-size:.8rem; color:#52665b; }
.sidebar-metric strong { display:block; font-size:1.6rem; line-height:1.3; color:#18342b; white-space:nowrap; }
.small-note { font-size:.8rem; color:#617368; line-height:1.55; }
.footer { border-top:1px solid #d5e1d2; padding-top:1.1rem; margin-top:2rem; font-size:.78rem; color:#63736a; }
@media(max-width:640px) {
 .block-container { padding:1.3rem 1rem; }
 .preview img { height:250px; }
 .result-card { padding:1rem; }
}
</style>
"""
st.markdown(CSS,unsafe_allow_html=True)

def show_image(image, alt):
    buffer=BytesIO()
    thumbnail=image.copy()
    thumbnail.thumbnail((1200,1200))
    thumbnail.convert("RGB").save(buffer,format="JPEG",quality=88)
    encoded=base64.b64encode(buffer.getvalue()).decode("ascii")
    st.markdown(f'<div class="preview"><img src="data:image/jpeg;base64,{encoded}" alt="{escape(alt)}"></div>',unsafe_allow_html=True)

def clear_upload():
    for key in ["file_id","analysis","overlay","preview","preview_error"]:
        st.session_state[key]=None
    st.session_state.upload_version+=1

def run_analysis(contents):
    # Import lazily: the landing page does not initialize TensorFlow or models.
    from utils.prediction import analyze_image
    with TemporaryDirectory(prefix="lemoscan_ui_") as directory:
        overlay_path=Path(directory)/"explanation.png"
        result=analyze_image(BytesIO(contents),overlay_path)
        overlay=None
        if result["status"]=="lemon_leaf_analysis" and result["gradcam"]["status"]=="available":
            try:
                with Image.open(overlay_path) as image:
                    overlay=image.convert("RGB").copy()
                    overlay.thumbnail((1200,1200))
            except (OSError,ValueError):
                result["gradcam"]={"status":"unavailable"}
                result["warnings"].append("Prediction completed, but the visual explanation could not be displayed.")
        # Keep UI state small; the rendered overlay is all the UI needs.
        result["gradcam"].pop("heatmap",None)
        result["gradcam"].pop("output_path",None)
        return result,overlay

def render_result(result,preview,overlay):
    status=result["status"]
    if status=="technical_error":
        st.error("We couldn't process this image. Please upload a valid JPG or PNG image.")
    elif status=="not_lemon_leaf":
        st.warning("Input not recognized as a lemon leaf")
        st.write("The validator did not recognize this image as a lemon leaf with sufficient confidence. Try another clear lemon-leaf image.")
        with st.expander("Input recognition details"):
            st.metric("Lemon-leaf score",f'{result["validator"]["lemon_score"]:.2%}')
            st.caption("This is the input validator's score, not disease confidence. The validator can be wrong.")
        return
    elif status=="analysis_error":
        stage=(result.get("error") or {}).get("stage")
        st.error("Input recognition is temporarily unavailable. Please try again later." if stage=="validator"
                 else "We couldn't complete disease analysis. Please try again later.")
        if result.get("validator",{} ) and result["validator"]["accepted_as_lemon_leaf"]:
            st.caption("Input recognition completed; no disease result is available.")
    elif status=="lemon_leaf_analysis":
        st.markdown("## Your leaf analysis")
        if result["legacy_uncertain"]:
            st.warning("Uncertain result — treat the leading prediction cautiously. The current uncertainty rule is provisional and not calibrated.")
        left,right=st.columns([1.15,1],gap="large")
        with left:
            label="Leading prediction · uncertain" if result["legacy_uncertain"] else "Predicted condition"
            st.markdown(f'<div class="result-card"><p>{label}</p><h2>{escape(result["predicted_class"])}</h2>'
                        f'<div class="score">{result["confidence"]:.2%}</div><p>Disease-model confidence</p></div>',unsafe_allow_html=True)
            st.caption("Model confidence is not a guarantee of correctness.")
        with right:
            st.markdown("### Top 3 predictions")
            for item in result["top_3"]:
                st.progress(float(item["probability"]),text=f'{item["class"]} · {item["probability"]:.2%}')
        st.markdown("### Why did the model focus here?")
        st.caption("Grad-CAM highlights image regions that contributed strongly to the model's prediction. It does not prove where a disease is located.")
        if result["gradcam"]["status"]=="available" and overlay is not None:
            if st.toggle("Show visual explanation",value=True,key="show_explanation"):
                original_col,cam_col=st.columns(2,gap="medium")
                with original_col:
                    st.markdown("**Original image**")
                    if preview is not None: show_image(preview,"Original uploaded image")
                with cam_col:
                    st.markdown(f'**Grad-CAM · {result["predicted_class"]}**')
                    show_image(overlay,"Grad-CAM explanation of the predicted class")
        else:
            st.info("Visual explanation unavailable for this analysis.")
        with st.expander("Analysis notes"):
            for warning in result.get("warnings",[]): st.write(warning)
            st.caption("The input validator and disease classifier have separate confidence scores. Neither is a calibrated guarantee.")
        report={key:value for key,value in result.items() if key!="gradcam"}
        report["gradcam"]={k:v for k,v in result["gradcam"].items() if k!="error"}
        st.download_button("Download analysis summary",json.dumps(report,indent=2),file_name="lemoscan_analysis.json",mime="application/json")
        return
    else:
        st.error("Analysis is unavailable. Please try another image.")
    if result.get("error"):
        with st.expander("Technical details"):
            st.text(result["error"].get("detail","No additional details available."))

def main():
    for key,value in {"upload_version":0,"file_id":None,"analysis":None,"overlay":None,"preview":None,"preview_error":None}.items():
        if key not in st.session_state: st.session_state[key]=value
    with st.sidebar:
        st.markdown("### LemoScan AI")
        st.caption("LEAF INTELLIGENCE · EXPLAINED")
        st.divider()
        st.markdown("## About LemoScan AI")
        st.write("A two-stage computer vision workflow for lemon-leaf analysis.")
        st.markdown("**Recognize** · MobileNetV3Small input validator\n\n**Classify** · InceptionV3 disease classifier\n\n**Explain** · Grad-CAM visual attribution")
        with st.expander("6 supported conditions",expanded=True):
            for name in CLASSES: st.write(name)
        st.divider()
        st.markdown("## Model performance")
        st.markdown('<div class="sidebar-metric"><span>Test accuracy</span><strong>95.44%</strong></div>'
                    '<div class="sidebar-metric"><span>Macro F1</span><strong>95.51%</strong></div>',
                    unsafe_allow_html=True)
        st.caption("Disease-model results on the held-out project test set: 307 images.")
        with st.expander("Methodology & limitations",expanded=False):
            st.write("The validator was tested on a constructed benchmark; broader real-world generalization remains unverified.")
            st.write("410 unresolved Category C/D similarity candidate pairs cross dataset splits. Complete biological-source independence is not proven.")
            st.write("Experimental research project; predictions should not replace expert agricultural diagnosis.")
        st.caption("Experimental AI · Six conditions · Visual explanations")
    st.markdown('<div class="hero"><div class="eyebrow">COMPUTER VISION / EXPLAINABLE AI</div>'
                '<h1>LemoScan AI</h1><p class="subtitle">Explainable AI for Lemon Leaf Disease Detection</p>'
                '<p>Check whether an image resembles a lemon leaf, explore six possible conditions, and see the regions that informed the model’s prediction.</p>'
                '<div class="chips"><span class="chip">Input recognition</span><span class="chip">6 leaf conditions</span>'
                '<span class="chip">Confidence ranking</span><span class="chip">Grad-CAM</span></div></div>',unsafe_allow_html=True)
    st.write("")
    upload_col,preview_col=st.columns([1,1.05],gap="large")
    with upload_col:
        st.markdown("## Start with a leaf")
        upload=st.file_uploader("Upload a lemon leaf image",type=["jpg","jpeg","png"],key=f'upload_{st.session_state.upload_version}',
                               help="JPG, JPEG or PNG. Your image is processed for this session.")
        contents=None if upload is None else upload.getvalue()
        file_id=None if contents is None else sha256(contents).hexdigest()
        if file_id!=st.session_state.file_id:
            st.session_state.file_id=file_id
            st.session_state.analysis=None
            st.session_state.overlay=None
            st.session_state.preview=None
            st.session_state.preview_error=None
            if contents is not None:
                try:
                    with Image.open(BytesIO(contents)) as image:
                        if image.width*image.height>40_000_000:
                            raise ValueError("Image exceeds the supported size.")
                        st.session_state.preview=image.convert("RGB").copy()
                        st.session_state.preview.thumbnail((1200,1200))
                except Exception:
                    st.session_state.preview_error="This file could not be previewed. Analyze it to check whether it is a supported image."
        st.caption("For best results")
        st.markdown("Use a clear photo with the leaf visible. Avoid excessive blur or extreme darkness; one main leaf usually works best.")
        action,clear=st.columns([1.5,1])
        with action:
            clicked=st.button("Analyze Leaf",type="primary",use_container_width=True,
                              disabled=contents is None or st.session_state.analysis is not None)
        with clear:
            st.button("Clear",on_click=clear_upload,use_container_width=True,disabled=contents is None)
        if clicked:
            with st.spinner("Checking the input and analyzing the leaf…"):
                try:
                    st.session_state.analysis,st.session_state.overlay=run_analysis(contents)
                except Exception:
                    st.session_state.analysis={"status":"analysis_error","validator":None,"message":"Analysis unavailable.",
                                               "error":{"stage":"application","detail":"The analysis service could not complete this request."}}
            st.rerun()
        if st.session_state.analysis is not None:
            st.caption("Analysis saved for this image. Upload another image or clear to start again.")
    with preview_col:
        st.markdown("### Image preview")
        if st.session_state.preview is not None:
            show_image(st.session_state.preview,"Uploaded image preview")
            st.caption("Your uploaded image")
        elif st.session_state.preview_error:
            st.warning(st.session_state.preview_error)
        else:
            st.markdown('<div class="preview" style="height:330px;display:flex;align-items:center;justify-content:center;text-align:center;">'
                        '<div><p style="font-size:1.1rem;font-weight:600;">Your next insight starts here.</p>'
                        '<p class="small-note">Upload a leaf photo to preview it.<br>Analysis runs only when you choose.</p></div></div>',unsafe_allow_html=True)
    if st.session_state.analysis is not None:
        st.divider()
        render_result(st.session_state.analysis,st.session_state.preview,st.session_state.overlay)
    st.write("")
    with st.expander("How LemoScan AI works"):
        st.markdown("1. **Image validation** — check that the file can be processed.\n"
                    "2. **Lemon-leaf recognition** — the input gate decides whether to continue.\n"
                    "3. **Disease classification** — rank six supported leaf conditions.\n"
                    "4. **Confidence ranking** — display the model's unchanged top-three probabilities.\n"
                    "5. **Grad-CAM explanation** — highlight regions contributing to the predicted class.")
    st.markdown('<div class="footer">LemoScan AI is an experimental AI project for educational and research purposes. '
                'Predictions should not replace expert agricultural diagnosis.</div>',unsafe_allow_html=True)

if __name__=="__main__":
    main()
