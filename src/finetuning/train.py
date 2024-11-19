"""
GLiNER Model Training Script

This script handles the training of a GLiNER model for named entity recognition.
It includes data loading, preprocessing, model configuration, and training setup.

Author: @sarrabenyahia 
Created: November 19, 2024
"""

import json
import random
import os
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset


@dataclass
class TrainingConfig:
    """Configuration class to store training parameters."""
    data_path: str = "./data/data.json"
    model_name: str = "almanach/camembert-bio-gliner-v0.1"
    output_dir: str = "models"
    train_split: float = 0.9
    num_steps: int = 500
    batch_size: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    others_lr: float = 1e-5
    others_weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: int = 2
    save_steps: int = 100
    save_total_limit: int = 10
    dataloader_workers: int = 0
    use_cpu: bool = False
    seed: int = 42


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(
    data_path: str,
    train_split: float,
    seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Load and split dataset into train and test sets.
    
    Args:
        data_path (str): Path to the JSON data file
        train_split (float): Proportion of data to use for training (0-1)
        seed (int, optional): Random seed for shuffling
        
    Returns:
        tuple: Training and test datasets
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    
    print(f'Dataset size: {len(data)}')
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_split)
    return data[:split_idx], data[split_idx:]


def setup_device() -> torch.device:
    """Setup and print CUDA device information.
    
    Returns:
        torch.device: Selected device (CPU or CUDA)
    """
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
    
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def calculate_num_epochs(num_steps: int, data_size: int, batch_size: int) -> int:
    """Calculate number of epochs based on desired steps and data size.
    
    Args:
        num_steps (int): Desired number of training steps
        data_size (int): Size of training dataset
        batch_size (int): Batch size
        
    Returns:
        int: Number of epochs to train
    """
    num_batches = data_size // batch_size
    return max(1, num_steps // num_batches)


def train_gliner_model(config: TrainingConfig) -> None:
    """Main training function for GLiNER model.
    
    Args:
        config (TrainingConfig): Training configuration parameters
    """
    # Set environment variables and seed
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    set_seed(config.seed)
    
    # Load and split data
    train_dataset, test_dataset = load_and_split_data(
        config.data_path,
        config.train_split,
        config.seed
    )
    
    # Setup device
    device = setup_device()
    
    # Initialize model
    model = GLiNER.from_pretrained(config.model_name)
    model.to(device)
    
    # Setup data collator
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )
    
    # Calculate number of epochs
    num_epochs = calculate_num_epochs(
        config.num_steps,
        len(train_dataset),
        config.batch_size
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        others_lr=config.others_lr,
        others_weight_decay=config.others_weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        focal_loss_alpha=config.focal_loss_alpha,
        focal_loss_gamma=config.focal_loss_gamma,
        num_train_epochs=num_epochs,
        eval_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        dataloader_num_workers=config.dataloader_workers,
        use_cpu=config.use_cpu,
        report_to="none",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    config = TrainingConfig(
        data_path="./data/data.json",
        output_dir="./models/BTB_gliner/",
        batch_size=16,
        num_steps=1000
    )
    train_gliner_model(config)

