"""
Checkpoint management with versioning and metadata
Supports automatic saving, loading, and version tracking
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class CheckpointMetadata:
    """Checkpoint metadata"""
    experiment_name: str
    epoch: int
    step: int
    metrics: Dict[str, float]
    model_config: Dict[str, Any]
    timestamp: str
    checksum: str
    is_best: bool = False
    description: str = ""


class CheckpointManager:
    """Manage model checkpoints with versioning"""
    
    def __init__(
        self,
        experiment_name: str,
        checkpoint_dir: str = "experiments/checkpoints",
        max_checkpoints: int = 5,
        keep_best: bool = True,
    ):
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        
        self.checkpoints: List[Dict[str, Any]] = []
        self.best_metric: Optional[float] = None
        self.best_metric_name: str = "loss"  # Default to loss (lower is better)
        
        self._load_checkpoint_index()
    
    def _load_checkpoint_index(self):
        """Load checkpoint index from disk"""
        index_file = self.checkpoint_dir / "index.json"
        
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.checkpoints = json.load(f)
    
    def _save_checkpoint_index(self):
        """Save checkpoint index to disk"""
        index_file = self.checkpoint_dir / "index.json"
        
        with open(index_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def save_checkpoint(
        self,
        model_state_dict,
        optimizer_state_dict: Optional[Dict] = None,
        scheduler_state_dict: Optional[Dict] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        description: str = "",
        is_best: bool = False,
    ) -> str:
        """Save a checkpoint with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save model state
        checkpoint_data = {
            "model_state_dict": model_state_dict,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": timestamp,
        }
        
        if optimizer_state_dict:
            checkpoint_data["optimizer_state_dict"] = optimizer_state_dict
        
        if scheduler_state_dict:
            checkpoint_data["scheduler_state_dict"] = scheduler_state_dict
        
        # Save checkpoint
        import torch
        torch.save(checkpoint_data, checkpoint_path)
        
        # Compute checksum
        checksum = self._compute_checksum(checkpoint_path)
        
        # Create metadata
        metadata = CheckpointMetadata(
            experiment_name=self.experiment_name,
            epoch=epoch,
            step=step,
            metrics=metrics or {},
            model_config=model_config or {},
            timestamp=timestamp,
            checksum=checksum,
            is_best=is_best,
            description=description,
        )
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Update index
        checkpoint_entry = {
            "name": checkpoint_name,
            "path": str(checkpoint_path),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "timestamp": timestamp,
            "checksum": checksum,
            "is_best": is_best,
        }
        
        self.checkpoints.append(checkpoint_entry)
        
        # Manage best checkpoint
        if is_best and self.keep_best:
            self._update_best_checkpoint(checkpoint_entry, metrics)
        
        # Prune old checkpoints
        self._prune_checkpoints()
        
        # Save index
        self._save_checkpoint_index()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _update_best_checkpoint(self, new_checkpoint: Dict, metrics: Dict[str, float]):
        """Update best checkpoint based on metric"""
        if self.best_metric is None:
            self.best_metric = metrics.get(self.best_metric_name, 0.0)
            new_checkpoint["is_best"] = True
            return
        
        current_metric = metrics.get(self.best_metric_name, 0.0)
        
        # For loss, lower is better
        if self.best_metric_name == "loss":
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                # Mark previous best as not best
                for cp in self.checkpoints:
                    cp["is_best"] = False
                new_checkpoint["is_best"] = True
        # For accuracy, higher is better
        else:
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                for cp in self.checkpoints:
                    cp["is_best"] = False
                new_checkpoint["is_best"] = True
    
    def _prune_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp (newest first)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        # Always keep best checkpoint
        best_checkpoint = None
        if self.keep_best:
            best_checkpoint = next(
                (cp for cp in sorted_checkpoints if cp["is_best"]),
                None
            )
        
        # Keep max_checkpoints most recent
        checkpoints_to_keep = sorted_checkpoints[:self.max_checkpoints]
        
        # Ensure best checkpoint is kept
        if best_checkpoint and best_checkpoint not in checkpoints_to_keep:
            checkpoints_to_keep[-1] = best_checkpoint
        
        # Remove old checkpoints
        checkpoints_to_remove = [
            cp for cp in self.checkpoints
            if cp not in checkpoints_to_keep
        ]
        
        for cp in checkpoints_to_remove:
            cp_path = Path(cp["path"])
            if cp_path.exists():
                cp_path.unlink()
            
            metadata_path = cp_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
            print(f"Removed old checkpoint: {cp['name']}")
        
        self.checkpoints = checkpoints_to_keep
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load a checkpoint"""
        import torch
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        
        result = {
            "model_state_dict": checkpoint_data["model_state_dict"],
            "epoch": checkpoint_data.get("epoch", 0),
            "step": checkpoint_data.get("step", 0),
            "metrics": checkpoint_data.get("metrics", {}),
        }
        
        if load_optimizer and "optimizer_state_dict" in checkpoint_data:
            result["optimizer_state_dict"] = checkpoint_data["optimizer_state_dict"]
        
        if "scheduler_state_dict" in checkpoint_data:
            result["scheduler_state_dict"] = checkpoint_data["scheduler_state_dict"]
        
        # Load metadata if available
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                result["metadata"] = json.load(f)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        
        return result
    
    def load_best_checkpoint(self):
        """Load the best checkpoint"""
        best_checkpoint = next(
            (cp for cp in self.checkpoints if cp["is_best"]),
            None
        )
        
        if best_checkpoint is None:
            raise ValueError("No best checkpoint found")
        
        return self.load_checkpoint(best_checkpoint["path"])
    
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        if not self.checkpoints:
            raise ValueError("No checkpoints found")
        
        latest_checkpoint = max(
            self.checkpoints,
            key=lambda x: x["timestamp"]
        )
        
        return self.load_checkpoint(latest_checkpoint["path"])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return self.checkpoints
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific checkpoint"""
        for cp in self.checkpoints:
            if str(cp["path"]) == checkpoint_path or cp["name"] == checkpoint_path:
                return cp
        return None
    
    def delete_checkpoint(self, checkpoint_path: str):
        """Delete a specific checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint_path.unlink()
        
        metadata_path = checkpoint_path.with_suffix('.json')
        if metadata_path.exists():
            metadata_path.unlink()
        
        # Remove from index
        self.checkpoints = [
            cp for cp in self.checkpoints
            if cp["path"] != str(checkpoint_path)
        ]
        
        self._save_checkpoint_index()
        
        print(f"Checkpoint deleted: {checkpoint_path}")
