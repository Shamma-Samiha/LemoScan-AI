import os
import uuid
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
RESULT_FOLDER = os.path.join("static", "results")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    """
    Check whether uploaded file is an allowed image type.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    """
    Home page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle image upload and prediction.
    """
    if "image" not in request.files:
        return render_template("index.html", error="No image file selected.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", error="Please upload a lemon leaf image.")

    if file and allowed_file(file.filename):
        from utils.prediction import predict_disease
        from utils.xai import generate_gradcam

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        # The prediction helper returns the best result, alternatives, and warning.
        predicted_class, confidence, top_predictions, warning = predict_disease(
            image_path
        )
        image_url = url_for("static", filename=f"uploads/{filename}")

        # Use a unique name so repeated uploads do not show an old cached result.
        image_name = os.path.splitext(filename)[0]
        gradcam_filename = f"{image_name}_gradcam_{uuid.uuid4().hex[:8]}.jpg"
        gradcam_output_path = os.path.join(
            app.config["RESULT_FOLDER"], gradcam_filename
        )
        generate_gradcam(image_path, gradcam_output_path)
        gradcam_url = url_for("static", filename=f"results/{gradcam_filename}")

        return render_template(
            "result.html",
            image_path=image_url,
            predicted_class=predicted_class,
            confidence=confidence,
            top_predictions=top_predictions,
            warning=warning,
            gradcam_path=gradcam_url,
        )

    return render_template("index.html", error="Only PNG, JPG, and JPEG images are allowed.")


if __name__ == "__main__":
    app.run(debug=True)
