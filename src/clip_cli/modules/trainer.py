"""
training module for CLIP
Supports mixed precision, gradient accumulation, learning rate scheduling, and comprehensive logging
"""

import clip
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from ..core.config import TrainingConfig, ModelConfig
from ..core.logger import ExperimentLogger
from ..core.checkpoint import CheckpointManager


class CLIPTrainer:
    """CLIP trainer with advanced features"""
    
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: TrainingConfig = None,
        model_config: ModelConfig = None,
        logger: Optional[ExperimentLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: str = "auto",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.model_config = model_config or ModelConfig()
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Freeze encoders if specified
        if self.model_config.freeze_image_encoder:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        
        if self.model_config.freeze_text_encoder:
            for param in self.model.transformer.parameters():
                param.requires_grad = False
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer.lower() == "adamw":
            optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "adam":
            optimizer = Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_epochs * len(self.train_loader),
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=(self.config.epochs - self.config.warmup_epochs) * len(self.train_loader),
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs * len(self.train_loader)],
        )
        
        return scheduler
    
    def clip_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP contrastive loss (InfoNCE)
        
        Args:
            image_features: Normalized image features [batch_size, dim]
            text_features: Normalized text features [batch_size, dim]
            logit_scale: Learnable temperature parameter
        
        Returns:
            Loss value
        """
        # Compute similarity matrix
        logits_per_image = logit_scale * (image_features @ text_features.T)
        logits_per_text = logits_per_image.T
        
        # Ground truth labels (diagonal)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Cross-entropy loss
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i + loss_t) / 2
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}",
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["images"].to(self.device)
            texts = clip.tokenize(batch["texts"], truncate=True).to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    loss = self.clip_loss(
                        image_features,
                        text_features,
                        self.model.logit_scale.exp(),
                    )
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
            else:
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                loss = self.clip_loss(
                    image_features,
                    text_features,
                    self.model.logit_scale.exp(),
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                
                if self.logger:
                    self.logger.log_metric("train/loss", avg_loss, step=self.global_step)
                    self.logger.log_metric("train/lr", lr, step=self.global_step)
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return {"loss": avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch in progress_bar:
            images = batch["images"].to(self.device)
            texts = clip.tokenize(batch["texts"], truncate=True).to(self.device)
            
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            loss = self.clip_loss(
                image_features,
                text_features,
                self.model.logit_scale.exp(),
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return {"loss": avg_loss}
    
    def train(self):
        """Main training loop"""
        if self.logger:
            self.logger.info("Starting training...")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Epochs: {self.config.epochs}")
            self.logger.info(f"Batch size: {self.train_loader.batch_size}")
            self.logger.info(f"Learning rate: {self.config.learning_rate}")
            self.logger.info(f"Mixed precision: {self.config.mixed_precision}")
        
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if self.val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate()
            
            # Log metrics
            if self.logger:
                self.logger.log_metrics(train_metrics, epoch=epoch + 1)
                if val_metrics:
                    self.logger.log_metrics(val_metrics, epoch=epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                is_best = val_metrics.get("loss", train_metrics["loss"]) < best_loss
                
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_checkpoint(
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict(),
                        scheduler_state_dict=self.scheduler.state_dict(),
                        epoch=epoch + 1,
                        step=self.global_step,
                        metrics={**train_metrics, **val_metrics},
                        model_config=self.model_config.__dict__,
                        description=f"Epoch {epoch + 1}",
                        is_best=is_best,
                    )
                
                # Update best metric
                current_loss = val_metrics.get("loss", train_metrics["loss"])
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    if self.logger:
                        self.logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                    break
        
        if self.logger:
            self.logger.info("Training completed")
