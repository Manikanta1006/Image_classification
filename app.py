import json
from pathlib import Path

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

from src.model import SimpleCNN


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "simple_cnn.pth"
CLASSES_PATH = ROOT_DIR / "models" / "classes.json"
IMAGE_SIZE = 224


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@st.cache_resource
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


st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("Image Classifier")

model, classes = load_model()

if model is None:
    st.warning("Train the model first so `models/simple_cnn.pth` and class names are available.")
    st.code("python src/train.py", language="bash")
else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

        label, confidence, probabilities = predict(image, model, classes)
        st.subheader(label)
        st.write(f"Confidence: {confidence:.2%}")

        scores = {class_name: float(probabilities[index]) for index, class_name in enumerate(classes)}
        st.bar_chart(scores)
