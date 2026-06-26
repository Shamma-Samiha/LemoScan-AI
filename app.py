import os
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

        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(image_path)

        predicted_class, confidence = predict_disease(image_path)
        image_url = url_for("static", filename=f"uploads/{filename}")

        return render_template(
            "result.html",
            image_path=image_url,
            predicted_class=predicted_class,
            confidence=round(confidence, 2)
        )

    return render_template("index.html", error="Only PNG, JPG, and JPEG images are allowed.")


if __name__ == "__main__":
    app.run(debug=True)
