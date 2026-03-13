# Face Disease Classification

A CNN-based image classifier that detects five facial skin conditions: **Acne**, **Actinic Keratosis**, **Basal Cell Carcinoma**, **Eczema**, and **Rosacea**.

## Project Structure

```
face_diseases/
├── main.py          # Entry point — training loop, data analysis
├── models.py        # FaceDiseaseCNN model and ClassificationLoss
├── train.py         # train() and evaluate() functions with TensorBoard logging
├── utils.py         # Dataset class, data loading, augmentation, analysis plots
├── runs.txt         # Training run notes
├── pyproject.toml   # Python dependencies (managed with uv)
├── data/            # Dataset (auto-downloaded from Kaggle on first run)
│   ├── train/       # Training images organized by class
│   │   ├── Acne/
│   │   ├── Actinic Keratosis/
│   │   ├── Basal Cell Carcinoma/
│   │   ├── Eczemaa/
│   │   └── Rosacea/
│   └── testing/     # Test images (same class subfolders)
└── logs/            # TensorBoard log files (generated during training)
```

## Setup

```bash
# Create virtual environment and install dependencies
uv sync
```

The dataset is downloaded automatically from Kaggle on first run. You need a Kaggle API key configured (`~/.kaggle/kaggle.json`).

## Running

By default, `main.py` runs both `analyze_data()` and `train_model()`:

```bash
uv run python main.py
```

### `analyze_data()`

Prints the class distribution (number of images and percentage per class) for both the **train** and **testing** splits.

### `train_model()`

Trains the CNN model and logs loss/accuracy to TensorBoard.

Parameters (adjustable in `main.py`):
- `epochs=50`
- `batch_size=64`
- `learning_rate=0.0001`

### View training logs

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 to view training/validation loss and accuracy curves.
