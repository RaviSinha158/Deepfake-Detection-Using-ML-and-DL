import os
import sys
from flask import Flask, render_template, request

# Allow importing backend
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from backend.predict import predict_image

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            # Predict (NOW MATCHES BACKEND)
            result, confidence, preview_name = predict_image(save_path)

            if preview_name:
                image_path = f"uploads/{preview_name}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
