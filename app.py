import json
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as transforms
from flask import Flask, request
from PIL import Image

from src.model import SimpleCNN


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "simple_cnn.pth"
CLASSES_PATH = ROOT_DIR / "models" / "classes.json"
IMAGE_SIZE = 224
app = Flask(__name__)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model():
    if not MODEL_PATH.exists():
        return None, []

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        classes = checkpoint.get("classes", [])
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
        classes = []

    if not classes and CLASSES_PATH.exists():
        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            classes = json.load(file)

    if not classes:
        return None, []

    model = SimpleCNN(num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


def predict(image, model, classes):
    image_tensor = get_transform()(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)
    return classes[predicted_index.item()], confidence.item(), probabilities


@app.route("/", methods=["GET", "POST"])
def index():
    model, classes = load_model()
    message = ""
    prediction_html = ""

    if model is None:
        message = "Train the model first: python src/train.py"
    elif request.method == "POST":
        uploaded_file = request.files.get("image")
        if not uploaded_file or uploaded_file.filename == "":
            message = "Choose an image first."
        else:
            image = Image.open(BytesIO(uploaded_file.read()))
            label, confidence, probabilities = predict(image, model, classes)
            rows = "".join(
                f"<li><strong>{class_name}</strong>: {float(probabilities[index]):.2%}</li>"
                for index, class_name in enumerate(classes)
            )
            prediction_html = f"""
                <section class="result">
                    <h2>{label}</h2>
                    <p>Confidence: {confidence:.2%}</p>
                    <ul>{rows}</ul>
                </section>
            """

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Image Classifier</title>
        <style>
            body {{
                margin: 0;
                min-height: 100vh;
                font-family: Arial, sans-serif;
                background: #f4f7fb;
                color: #152238;
                display: grid;
                place-items: center;
            }}
            main {{
                width: min(680px, calc(100vw - 32px));
                background: #ffffff;
                border: 1px solid #d9e1ec;
                border-radius: 8px;
                padding: 32px;
                box-shadow: 0 16px 40px rgba(21, 34, 56, 0.08);
            }}
            h1 {{ margin-top: 0; }}
            form {{
                display: flex;
                gap: 12px;
                align-items: center;
                flex-wrap: wrap;
            }}
            input, button {{
                font: inherit;
            }}
            button {{
                border: 0;
                border-radius: 6px;
                padding: 10px 16px;
                background: #2563eb;
                color: white;
                cursor: pointer;
            }}
            .message {{
                margin-top: 18px;
                color: #b45309;
                font-weight: 600;
            }}
            .result {{
                margin-top: 24px;
                border-top: 1px solid #d9e1ec;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <main>
            <h1>Image Classifier</h1>
            <form method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*">
                <button type="submit">Predict</button>
            </form>
            <p class="message">{message}</p>
            {prediction_html}
        </main>
    </body>
    </html>
    """


@app.route("/health")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
