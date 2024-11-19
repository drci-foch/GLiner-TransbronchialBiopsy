# 🚑 GLiner-TransbronchialBiopsy

A medical named entity recognition system using fine-tuned GLiNER models, with an interactive dashboard for medical text annotation.

## 📋 Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)

## ⚙️ Installation

```bash
# Create virtual environment
python -m venv gliner-env

# Activate virtual environment
# Windows:
gliner-env\Scripts\activate
# Unix/macOS:
source gliner-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Features

### Training Pipeline
- Custom training loop with cyclic learning rate scheduler
- Early stopping and gradient clipping
- L2 regularization (weight decay)
- Configurable hyperparameters
- Progress monitoring and model checkpointing

### Interactive Dashboard
- Real-time medical text annotation
- Entity highlighting with color coding
- Interactive statistics and visualizations
- Adjustable confidence threshold
- Entity distribution charts

## 💻 Usage

### Training
```bash
python train_overfit_gradient.py
```

Key configuration options in `TrainingConfig`:
- `data_path`: Path to training data
- `batch_size`: Training batch size
- `learning_rate`: Base learning rate
- `num_steps`: Total training steps

### Dashboard
```bash
streamlit run dashboard.py
```

The dashboard provides:
- Text input area for medical documents
- Color-coded entity highlighting
- Interactive confidence threshold adjustment
- Entity distribution visualization
- Detailed entity statistics

## 📊 Supported Entities
- Site
- Nombre Total De Fragments
- Nombre Total De Fragments Alvéolés
- Grade A/B
- Rejet Chronique
- Coloration C4d
- And more medical-specific entities

## 📈 Model Performance
Monitor training progress and evaluation metrics through:
- Loss tracking
- Early stopping metrics
- Evaluation steps
- Model checkpoints

## 🛠️ Configuration
Adjust model parameters through `TrainingConfig` in `train_overfit_gradient.py`:
```python
config = TrainingConfig(
    data_path="./data/data.json",
    output_dir="./models/custom_run",
    batch_size=8,
    num_steps=1000
)
```