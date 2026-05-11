# Image Classification AI/ML Project

This project trains a simple PyTorch image classifier and serves it with a Streamlit web app for deployment on Lightning AI.

## Folder Structure

```text
Image_classification/
|-- app.py                  # Streamlit prediction app
|-- data/
|   |-- train/              # Training images, one folder per class
|   `-- val/                # Optional validation images
|-- models/                 # Trained model weights and class names
|-- src/
|   |-- dataset.py
|   |-- model.py
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Prepare Data

Organize images into class folders:

```text
data/train/cats/
data/train/dogs/
```

Optional validation data should use the same class folder names:

```text
data/val/cats/
data/val/dogs/
```

## 3. Train the Model

```bash
python src/train.py
```

The training script saves:

```text
models/simple_cnn.pth
models/classes.json
```

You can also change training settings:

```bash
python src/train.py --epochs 20 --batch-size 16 --learning-rate 0.0005
```

## 4. Run the Web App

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```

Upload an image in the app to get the predicted class and confidence.

## Lightning AI Deployment

1. Create or open a Lightning AI Studio.
2. Upload this folder or clone it from GitHub.
3. In the Studio terminal, run:

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py --server.address 0.0.0.0 --server.port 8080
```

4. Open the public or preview URL for port `8080`.
