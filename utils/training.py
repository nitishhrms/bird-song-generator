"""
Training Utilities

Common training functions for all models.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    Base trainer class with common functionality
    
    Args:
        model: PyTorch model
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    def __init__(self, model, device, checkpoint_dir='checkpoints', log_dir='logs'):
        self.model = model
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def save_checkpoint(self, filename, **kwargs):
        """
        Save model checkpoint
        
        Args:
            filename: Checkpoint filename
            **kwargs: Additional items to save
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            **kwargs
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename, **kwargs):
        """
        Load model checkpoint
        
        Args:
            filename: Checkpoint filename
            **kwargs: Additional items to load (e.g., optimizers)
        
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def log_scalar(self, tag, value, step=None):
        """Log scalar value to TensorBoard"""
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag, image, step=None):
        """Log image to TensorBoard"""
        step = step if step is not None else self.global_step
        self.writer.add_image(tag, image, step)
    
    def log_audio(self, tag, audio, sample_rate=22050, step=None):
        """Log audio to TensorBoard"""
        step = step if step is not None else self.global_step
        self.writer.add_audio(tag, audio, step, sample_rate=sample_rate)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()


class ExperimentLogger:
    """
    Logger for tracking experiments
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
    """
    def __init__(self, experiment_name, log_dir='experiments'):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.experiment_dir / 'log.json'
        self.log_data = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'config': {},
            'metrics': []
        }
    
    def log_config(self, config):
        """Log experiment configuration"""
        self.log_data['config'] = config
        self._save_log()
    
    def log_metrics(self, epoch, metrics):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.log_data['metrics'].append(entry)
        self._save_log()
    
    def _save_log(self):
        """Save log to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_experiment_dir(self):
        """Get experiment directory path"""
        return self.experiment_dir


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' or 'max' (whether lower or higher is better)
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """
        Check if training should stop
        
        Args:
            score: Current validation score
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_optimizer(model, optimizer_name='adam', lr=0.0002, **kwargs):
    """
    Get optimizer for model
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='step', **kwargs):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Additional scheduler arguments
    
    Returns:
        Scheduler
    """
    if scheduler_name.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print("Training utilities module")
    print("=" * 50)
    print("\nAvailable utilities:")
    print("  - Trainer: Base trainer with checkpointing and logging")
    print("  - ExperimentLogger: Track experiments with JSON logs")
    print("  - EarlyStopping: Stop training when validation plateaus")
    print("  - get_optimizer() / get_scheduler()")
    print("  - count_parameters() / set_seed()")
