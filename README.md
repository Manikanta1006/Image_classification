# Image Classification AI/ML Project

This project trains a simple PyTorch image classifier and serves it with a Flask web app for Lightning AI.

## Run on Lightning AI

In the Lightning AI terminal:

```bash
cd /teamspace/studios/this_studio/Image_classification2
pip install -r requirements.txt
python src/train.py
python app.py
```

The app starts on port `8501`.

Open this Lightning Web UI URL:

```text
https://lightning.ai/veeramanikanta386/deploy-model-project/studios/favourable-violet-myp7/web-ui?port=8501
```

If port `8501` is busy, run:

```bash
python -c "from app import app; app.run(host='0.0.0.0', port=8502)"
```

Then open the same URL with `port=8502`.

## Data Layout

Put images in class folders:

```text
data/train/cats/
data/train/dogs/
```

Optional validation data:

```text
data/val/cats/
data/val/dogs/
```

## Files

```text
app.py                  Flask prediction app
src/train.py            Training script
src/model.py            CNN model
src/dataset.py          Data loaders
requirements.txt        Python dependencies
```
