"""
Configuration management for CLIP experiments
Supports YAML configs with validation
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "ViT-B-32"
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = "data"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    image_size: int = 224
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 1
    early_stopping_patience: int = 5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    mixed_precision: bool = True
    
    # Granular training control
    train_vision_encoder: bool = True
    train_text_encoder: bool = True
    train_image_projection: bool = True
    train_text_projection: bool = True
    train_logit_scale: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    batch_size: int = 64
    num_workers: int = 4
    metrics: List[str] = field(default_factory=lambda: [
        "recall@1", "recall@5", "recall@10",
        "precision@1", "precision@5",
        "mean_rank", "mrr"
    ])


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    name: str = "clip_experiment"
    description: str = ""
    seed: int = 42
    device: str = "auto"
    mixed_precision: bool = True
    distributed: bool = False
    world_size: int = 1
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


class ConfigManager:
    """Manage experiment configurations"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            if config_path.name in ["default.yaml", "base.yaml"]:
                return self._create_default_config(config_path)
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: ExperimentConfig, config_path: str):
        """Save configuration to YAML file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"Configuration saved to {config_path}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig"""
        # Handle nested experiment section
        if 'experiment' in config_dict and isinstance(config_dict['experiment'], dict):
            experiment_dict = config_dict.pop('experiment')
            config_dict.update(experiment_dict)
        
        # Handle nested configs
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        if 'evaluation' in config_dict and isinstance(config_dict['evaluation'], dict):
            config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        return ExperimentConfig(**config_dict)
    
    def _create_default_config(self, config_path: Path) -> ExperimentConfig:
        """Create and save default configuration"""
        config = ExperimentConfig()
        self.save_config(config, config_path)
        return config
    
    def create_experiment_config(
        self,
        experiment_name: str,
        base_config: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """Create a new experiment configuration"""
        if base_config:
            config = self.load_config(base_config)
        else:
            config = ExperimentConfig()
        
        config.name = experiment_name
        
        if overrides:
            self._apply_overrides(config, overrides)
        
        # Save experiment-specific config
        experiment_config_path = self.config_dir / f"{experiment_name}.yaml"
        self.save_config(config, experiment_config_path)
        
        return config
    
    def _apply_overrides(self, config: ExperimentConfig, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.evaluation, key):
                setattr(config.evaluation, key, value)


def get_default_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig()
