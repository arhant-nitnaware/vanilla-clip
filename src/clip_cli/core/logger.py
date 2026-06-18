"""
Experiment tracking and logging system
Supports TensorBoard, WandB, and file logging
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """Comprehensive experiment logging system"""
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "experiments/logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # File logging
        self._setup_file_logger()
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(str(tb_dir))
            print(f"TensorBoard logging enabled: {tb_dir}")
        
        # WandB
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE and wandb_project:
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=experiment_name,
                dir=str(self.log_dir),
                reinit=True
            )
            print(f"WandB logging enabled: {wandb_project}")
        
        # Metrics storage
        self.metrics_history: Dict[str, list] = {}
        
        # Config storage
        self.config: Optional[Dict[str, Any]] = None
    
    def _setup_file_logger(self):
        """Setup file and console logging"""
        self.logger = logging.getLogger(f"clip_cli.{self.experiment_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.config = config
        config_file = self.log_dir / "config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {config_file}")
        
        if self.wandb_run:
            wandb.config.update(config)
    
    def log_metric(
        self,
        name: str,
        value: Union[float, int],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """Log a metric"""
        # Store in history
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        log_entry = {
            "value": float(value),
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat()
        }
        self.metrics_history[name].append(log_entry)
        
        # Log to file
        self.logger.debug(f"Metric {name}: {value} (step={step}, epoch={epoch})")
        
        # Log to TensorBoard
        if self.tb_writer:
            if step is not None:
                self.tb_writer.add_scalar(name, value, step)
            elif epoch is not None:
                self.tb_writer.add_scalar(name, value, epoch)
        
        # Log to WandB
        if self.wandb_run:
            wandb.log({name: value}, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """Log multiple metrics"""
        for name, value in metrics.items():
            self.log_metric(name, value, step, epoch)
    
    def log_image(self, tag: str, image, step: Optional[int] = None):
        """Log an image"""
        if self.tb_writer:
            self.tb_writer.add_image(tag, image, step)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """Log text"""
        if self.tb_writer:
            self.tb_writer.add_text(tag, text, step)
        
        if self.wandb_run:
            wandb.log({tag: text}, step=step)
    
    def log_histogram(self, tag: str, values, step: Optional[int] = None):
        """Log histogram"""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
        
        if self.wandb_run:
            wandb.log({tag: wandb.Histogram(values)}, step=step)
    
    def log_hyperparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """Log hyperparameters with metrics"""
        if self.tb_writer:
            self.tb_writer.add_hparams(hparam_dict, metric_dict)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def save_metrics(self):
        """Save metrics history to JSON"""
        metrics_file = self.log_dir / "metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def close(self):
        """Close all loggers"""
        self.save_metrics()
        
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        for handler in self.logger.handlers:
            handler.close()
    
    @contextmanager
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


def get_logger(
    experiment_name: str,
    log_dir: str = "experiments/logs",
    **kwargs
) -> ExperimentLogger:
    """Factory function to create logger"""
    return ExperimentLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        **kwargs
    )
