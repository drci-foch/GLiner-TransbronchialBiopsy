# GLiner-TransbronchialBiopsy

This project fine-tunes the GLiNER model for named entity recognition tasks using the Hugging Face Transformers library. Follow the steps below to set up and run the project.

## Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)

## Setup

### 1. Create a Virtual Environment

First, create a virtual environment to isolate your project's dependencies.

```bash
# Create a virtual environment
python -m venv gliner-env

# Activate the virtual environment
# On Windows
gliner-env\Scripts\activate
# On macOS/Linux
source gliner-env/bin/activate
```

### 2. Install poetry
If you haven't installed Poetry yet, you can do so by following the official installation guide [here](https://python-poetry.org/docs/).

### 3. Install Project Dependencies
Navigate to the root directory of your project (where the pyproject.toml file is located) and install the dependencies using Poetry.

### 4. Run the Training Script
Once all dependencies are installed, you can run the training script. Ensure that you are in the virtual environment and that the script has the correct permissions.
```bash
# Run the training script
python train.py --config path/to/your/config.yaml --log_dir path/to/save/models
```



