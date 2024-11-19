<div align="center">
  
# 🚑 GLiner-TransbronchialBiopsy

<img src="icon.png" width="500" height="500" alt="Medical NER System">

A specialized medical Named Entity Recognition (NER) system for analyzing transbronchial biopsy reports, powered by fine-tuned GLiNER models.

</div>

## 🎯 Project Overview

GLiner-TransbronchialBiopsy is designed specifically for extracting medical entities from transbronchial biopsy reports, with a focus on transplant rejection analysis. The system combines state-of-the-art NLP techniques with domain-specific medical knowledge.

## 🔍 Key Features

- **Specialized Medical NER**: Tailored for transbronchial biopsy reports
- **Interactive Annotation**: Real-time medical text processing
- **Comprehensive Entity Coverage**: Focuses on critical biopsy parameters
- **Performance Optimization**: Fine-tuned for medical domain accuracy

## 🔧 Technical Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB RAM minimum
- 2GB free disk space

## 📦 Installation

```bash
# Create and activate virtual environment
python -m venv gliner-env
source gliner-env/bin/activate  # Unix/macOS
gliner-env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🎓 Supported Medical Entities

| Entity Type | Description | Example |
|------------|-------------|---------|
| Site | Biopsy location | "LSD", "LM" |
| Fragment Count | Total fragments analyzed | "4 fragments" |
| Alveolar Count | Number of alveolar fragments | "3 fragments alvéolés" |
| Rejection Grade | A/B grading scale | "Grade A2" |
| Chronic Rejection | Presence indicators | "Rejet chronique minimal" |
| C4d Staining | Staining results | "C4d négatif" |

## 💻 Usage Guide

### Model Training

```python
from gliner_transbronchial import TrainingConfig

config = TrainingConfig(
    data_path="./data/biopsy_reports.json",
    output_dir="./models/production",
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=10
)

trainer.train(config)
```

### Interactive Dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

## 🔄 Development Workflow

1. Data Preparation
   - Report collection
   - Manual annotation
   - Quality assurance

2. Model Training
   - Hyperparameter optimization
   - Cross-validation
   - Error analysis

3. Evaluation
   - Performance metrics
   - Clinical validation
   - Error analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
