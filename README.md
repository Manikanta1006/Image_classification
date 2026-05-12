# Image Classification AI/ML Project

This project trains a simple PyTorch image classifier and serves it with a Streamlit web app for Lightning AI.

## Run on Lightning AI

In the Lightning AI terminal:

```bash
cd /teamspace/studios/this_studio/Image_classification2
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

The project includes `.streamlit/config.toml`, so Streamlit starts on port `8501` automatically.

Open this Lightning Web UI URL:

```text
https://lightning.ai/veeramanikanta386/deploy-model-project/studios/favourable-violet-myp7/web-ui?port=8501
```

If port `8501` is busy, run:

```bash
streamlit run app.py --server.port 8502
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
app.py                  Streamlit prediction app
src/train.py            Training script
src/model.py            CNN model
src/dataset.py          Data loaders
requirements.txt        Python dependencies
.streamlit/config.toml  Lightning/Streamlit server config
```
