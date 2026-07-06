from flask import Flask, render_template, request, jsonify
from backend.predict import predict_image
import os

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend"
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


# ==========================================================
# API ROUTE: /predict
# ==========================================================
# This endpoint receives the image from the frontend via POST,
# saves it temporarily, and passes it to the Vision Transformer.
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        })

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "No image selected"
        })

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # Pass the saved image path to our PyTorch backend
    result, confidence, heatmap_b64, embedding_b64, inference_time = predict_image(filepath)

    # Return the AI's classification, confidence, and the visual 
    # attention heatmap back to the frontend as a JSON response.
    return jsonify({

        "prediction": result,

        "confidence": confidence,
        
        "heatmap": heatmap_b64,
        
        "embedding": embedding_b64,
        
        "inference_time": inference_time

    })


if __name__ == "__main__":
    # Port 7860 is required by Hugging Face Spaces Cloud deployment
    app.run(host="0.0.0.0", port=7860, debug=False)