"""
GLiNER Model Training Script V0.3

This script provides a comprehensive implementation of GLiNER model training
with advanced features for handling overfitting and optimization.

Author: @sarrabenyahia
Date: November 19, 2024
License: MIT
"""

import json
import random
import os
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import datetime
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import KFold
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for GLiNER training with all hyperparameters."""
    
    # Data parameters
    data_path: str = "./data/data.json"
    model_name: str = "almanach/camembert-bio-gliner-v0.1"
    output_dir: str = "models"
    train_split: float = 0.9
    
    # Learning rate parameters
    base_learning_rate: float = 5e-6
    max_learning_rate: float = 2e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    
    # Cyclic learning rate parameters
    cycles: int = 3
    cycle_momentum: bool = True
    
    # Training parameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    num_steps: int = 1000
    warmup_steps: int = 100
    
    # Regularization
    dropout_rate: float = 0.2
    layer_norm_eps: float = 1e-5
    
    # Monitoring parameters
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.005
    eval_steps: int = 50
    
    # Loss parameters
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: int = 2
    
    # K-fold parameters
    n_folds: int = 5
    use_kfold: bool = True
    save_all_folds: bool = False  # Whether to save models from all folds

    # Other parameters
    seed: int = 42
    use_cpu: bool = False
    dataloader_workers: int = 2
    save_total_limit: int = 3


class KFoldTrainingResults:
    """Class to track and aggregate results across k-fold runs."""
    
    def __init__(self, n_folds: int):
        self.n_folds = n_folds
        self.fold_metrics = defaultdict(list)
        self.best_fold = None
        self.best_fold_metrics = None
        self.best_fold_loss = float('inf')
        
    def add_fold_result(self, fold: int, metrics: Dict[str, float]):
        """Add results from a single fold."""
        for metric, value in metrics.items():
            self.fold_metrics[metric].append(value)
            
        # Track best fold based on eval_loss
        if metrics['eval_loss'] < self.best_fold_loss:
            self.best_fold = fold
            self.best_fold_metrics = metrics
            self.best_fold_loss = metrics['eval_loss']
    
    def get_aggregate_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate aggregate statistics across all folds."""
        aggregates = {}
        for metric in self.fold_metrics.keys():
            values = np.array(self.fold_metrics[metric])
            aggregates[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        return aggregates



class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    
    def __init__(self, patience: int, threshold: float, mode: str = 'min'):
        """Initialize early stopping.
        
        Args:
            patience (int): Number of checks to wait
            threshold (float): Minimum change threshold
            mode (str): 'min' for loss, 'max' for metrics
        """
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, epoch: int, current_value: float) -> bool:
        """Check if training should stop.
        
        Args:
            epoch (int): Current epoch
            current_value (float): Current metric value
            
        Returns:
            bool: True if should stop
        """
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.threshold)
        else:
            improved = current_value > (self.best_value + self.threshold)
            
        if improved:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(f'Early stopping triggered. Best value: {self.best_value:.4f} at epoch {self.best_epoch}')
            return True
        return False


class CyclicLRScheduler:
    """Custom cyclic learning rate scheduler with warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        max_lr: float,
        min_lr: float,
        warmup_steps: int,
        cycles: int,
        total_steps: int
    ):
        """Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            base_lr: Base learning rate
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate 
            warmup_steps: Number of warmup steps
            cycles: Number of cycles
            total_steps: Total training steps
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.cycles = cycles
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self) -> float:
        """Update learning rate and return current value."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cyclic cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cycle_progress = progress * self.cycles
            cosine_decay = 0.5 * (1 + np.cos(np.pi * (cycle_progress % 1)))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
        return lr


class MetricsLogger:
    """Class to handle metrics logging to file."""
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """Initialize logger with directory and optional experiment name."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_prefix = f"{experiment_name}_" if experiment_name else ""
        self.log_file = self.log_dir / f"{experiment_prefix}training_log_{timestamp}.txt"
        
        # Initialize file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - Started at {timestamp}\n")
            f.write("-" * 80 + "\n\n")
    
    def log_config(self, config: Dict):
        """Log configuration parameters."""
        with open(self.log_file, 'a') as f:
            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    def log_metrics(self, metrics: Dict, fold: Optional[int] = None, step: Optional[int] = None):
        """Log metrics to file."""
        with open(self.log_file, 'a') as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[Fold {fold}] " if fold is not None else ""
            step_str = f"Step {step} - " if step is not None else ""
            f.write(f"{prefix}{step_str}{timestamp}:\n")
            f.write(json.dumps(metrics, indent=2) + "\n\n")
    
    def log_fold_summary(self, fold: int, metrics: Dict):
        """Log summary metrics for a fold."""
        with open(self.log_file, 'a') as f:
            f.write(f"Fold {fold} Summary:\n")
            f.write(json.dumps(metrics, indent=2) + "\n")
            f.write("-" * 80 + "\n\n")
    
    def log_final_summary(self, aggregate_metrics: Dict):
        """Log final summary of all folds."""
        with open(self.log_file, 'a') as f:
            f.write("Final K-Fold Cross Validation Results:\n")
            f.write(json.dumps(aggregate_metrics, indent=2) + "\n")
            f.write("-" * 80 + "\n\n")


class ImprovedTrainer(Trainer):
    """Enhanced Trainer with additional features."""
    
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        metrics_logger: MetricsLogger,
        early_stopping: EarlyStopping,
        scheduler: CyclicLRScheduler,
        max_grad_norm: float,
        current_fold: Optional[int] = None,
    ):
        """Initialize trainer with custom components."""
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        self.metrics_logger = metrics_logger
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.current_fold = current_fold
        self.best_model_state = None
    
    def log(self, logs: Dict[str, float]) -> None:
        """Override log method to include file logging."""
        super().log(logs)
        
        # Log metrics to file
        self.metrics_logger.log_metrics(
            metrics=logs,
            fold=self.current_fold,
            step=self.state.global_step
        )
        
    def training_step(self, model, inputs):
        """Enhanced training step with gradient clipping."""
        loss = super().training_step(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.max_grad_norm
            )
            self.scheduler.step()
            
        return loss
        
    def evaluate(self, *args, **kwargs):
        """Enhanced evaluation with early stopping check."""
        metrics = super().evaluate(*args, **kwargs)
        
        if self.early_stopping(
            self.state.epoch,
            metrics['eval_loss']
        ):
            self.model.stop_training = True
            
        return metrics
    
    def save_best_model(self):
        """Save the best model state."""
        self.best_model_state = {
            k: v.cpu().clone() for k, v in self.model.state_dict().items()
        }
        
    def load_best_model(self):
        """Load the best model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(
    data_path: str,
    train_split: float,
    seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict]]:
    """Load and split dataset."""
    logger.info(f'Loading data from {data_path}')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f'Dataset size: {len(data)}')
    
    if seed is not None:
        random.seed(seed)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_split)
    return data[:split_idx], data[split_idx:]


def create_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with parameter-specific settings."""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.base_learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )


def setup_device() -> torch.device:
    """Setup and verify CUDA device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f'Using GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    else:
        device = torch.device('cpu')
        logger.info('Using CPU')
    
    return device


def train_gliner_kfold(config: TrainingConfig) -> KFoldTrainingResults:
    """Main training function with k-fold cross-validation and logging."""
    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        log_dir=os.path.join(config.output_dir, 'logs'),
        experiment_name=Path(config.output_dir).name
    )
    
    # Log configuration
    metrics_logger.log_config(config.__dict__)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(config.seed)
    
    # Load all data
    with open(config.data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Setup device
    device = setup_device()
    
    # Initialize k-fold cross validation
    kfold_splitter = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    
    # Initialize results tracker
    results = KFoldTrainingResults(config.n_folds)
    
    # Run training for each fold
    for fold, (train_idx, val_idx) in enumerate(kfold_splitter.split(all_data)):
        logger.info(f'\nStarting training for fold {fold + 1}/{config.n_folds}')
        metrics_logger.log_metrics(
            {'message': f'Starting training for fold {fold + 1}/{config.n_folds}'},
            fold=fold
        )
        
        # Split data for this fold
        train_dataset = [all_data[i] for i in train_idx]
        val_dataset = [all_data[i] for i in val_idx]
        
        # Initialize model for this fold
        model = GLiNER.from_pretrained(config.model_name)
        
        # Configure dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = config.dropout_rate
        
        model.to(device)
        
        # Create optimizer
        optimizer = create_optimizer(model, config)
        
        # Create scheduler
        scheduler = CyclicLRScheduler(
            optimizer=optimizer,
            base_lr=config.base_learning_rate,
            max_lr=config.max_learning_rate,
            min_lr=config.min_learning_rate,
            warmup_steps=config.warmup_steps,
            cycles=config.cycles,
            total_steps=config.num_steps
        )
        
        # Create data collator
        data_collator = DataCollator(
            model.config,
            data_processor=model.data_processor,
            prepare_labels=True
        )
        
        # Calculate steps per epoch for this fold
        steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
        num_train_epochs = (config.num_steps + steps_per_epoch - 1) // steps_per_epoch
        
        # Setup training arguments for this fold
        fold_output_dir = os.path.join(config.output_dir, f'fold_{fold}')
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            learning_rate=config.base_learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_grad_norm=config.max_grad_norm,
            num_train_epochs=num_train_epochs,
            max_steps=config.num_steps,
            warmup_steps=config.warmup_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_steps=config.eval_steps,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            focal_loss_alpha=config.focal_loss_alpha,
            focal_loss_gamma=config.focal_loss_gamma,
            dataloader_num_workers=config.dataloader_workers,
            use_cpu=config.use_cpu,
            report_to="none",
            remove_unused_columns=False,
            logging_steps=10,
            logging_first_step=True,
            seed=config.seed + fold,  # Different seed for each fold
            data_seed=config.seed + fold,
            tf32=True,
        )
        
        # Create early stopping handler
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            threshold=config.early_stopping_threshold
        )
        
        # Initialize trainer for this fold
        trainer = ImprovedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=model.data_processor.transformer_tokenizer,
            data_collator=data_collator,
            metrics_logger=metrics_logger,
            early_stopping=early_stopping,
            scheduler=scheduler,
            max_grad_norm=config.max_grad_norm,
            current_fold=fold,
        )
        
        try:
            # Train the model for this fold
            trainer.train()
            
            # Load best model for this fold
            trainer.load_best_model()
            
            # Evaluate the model
            metrics = trainer.evaluate()
            results.add_fold_result(fold, metrics)
            
            # Log fold summary
            metrics_logger.log_fold_summary(fold, metrics)
            
            # Save models if configured
            if config.save_all_folds:
                trainer.save_model(fold_output_dir)
                metrics_logger.log_metrics(
                    {'message': f'Saved model for fold {fold + 1}'},
                    fold=fold
                )
            
            if fold == results.best_fold:
                best_model_dir = os.path.join(config.output_dir, 'best_fold_model')
                trainer.save_model(best_model_dir)
                metrics_logger.log_metrics(
                    {'message': f'Saved best model (fold {fold + 1})'},
                    fold=fold
                )
            
        except Exception as e:
            error_msg = f'Training failed for fold {fold + 1} with error: {str(e)}'
            logger.error(error_msg)
            metrics_logger.log_metrics({'error': error_msg}, fold=fold)
            continue
    
    # Log final results
    aggregate_metrics = results.get_aggregate_metrics()
    metrics_logger.log_final_summary(aggregate_metrics)
    
    return results


# Example usage
if __name__ == "__main__":
    config = TrainingConfig(
        data_path="./data/data.json",
        output_dir="./models/kfold_run",
        batch_size=8,
        num_steps=1000,
        base_learning_rate=5e-6,
        max_learning_rate=2e-5,
        cycles=3,
        gradient_accumulation_steps=4,
        eval_steps=25,
        n_folds=5,
        use_kfold=True
    )
    
    results = train_gliner_kfold(config)