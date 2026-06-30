"""
training module for CLIP
Supports mixed precision, gradient accumulation, learning rate scheduling, and comprehensive logging
"""

import clip
import torch
import torch.nn.functional as F
import numpy as np
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
        # Ensure model is in float32 for training stability (necessary since clip.load defaults to float16 on CUDA)
        self.model.float()
        
        # Get image projection parameter(s)/module
        image_projection_params = []
        if hasattr(self.model, 'visual'):
            if hasattr(self.model.visual, 'proj') and self.model.visual.proj is not None:
                # ViT: proj is a Parameter
                image_projection_params = [self.model.visual.proj]
            elif hasattr(self.model.visual, 'attnpool') and hasattr(self.model.visual.attnpool, 'c_proj'):
                # ResNet: attnpool.c_proj is a Linear module
                image_projection_params = list(self.model.visual.attnpool.c_proj.parameters())
        
        # Granular training control
        # Vision encoder
        if not self.config.train_vision_encoder:
            for param in self.model.visual.parameters():
                param.requires_grad = False
        
        # Image projection
        if image_projection_params:
            for param in image_projection_params:
                param.requires_grad = self.config.train_image_projection
            
            # Re-initialize only if explicitly training from scratch (not pretrained)
            if self.config.train_image_projection and not self.model_config.pretrained:
                for param in image_projection_params:
                    if param.dim() > 1:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
        
        # Text encoder
        if not self.config.train_text_encoder:
            # Freeze transformer blocks
            if hasattr(self.model, 'transformer'):
                for param in self.model.transformer.parameters():
                    param.requires_grad = False
            # Freeze token embeddings
            if hasattr(self.model, 'token_embedding'):
                for param in self.model.token_embedding.parameters():
                    param.requires_grad = False
            # Freeze positional embeddings
            if hasattr(self.model, 'positional_embedding') and self.model.positional_embedding is not None:
                self.model.positional_embedding.requires_grad = False
            # Freeze final layer norm
            if hasattr(self.model, 'ln_final'):
                for param in self.model.ln_final.parameters():
                    param.requires_grad = False
        
        # Text projection
        if hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
            self.model.text_projection.requires_grad = self.config.train_text_projection
            # Re-initialize text projection if it's trainable and not pretrained
            if self.config.train_text_projection and not self.model_config.pretrained:
                torch.nn.init.xavier_uniform_(self.model.text_projection)
        
        # Logit scale - initialize to a reasonable value if trainable
        if hasattr(self.model, 'logit_scale') and self.model.logit_scale is not None:
            self.model.logit_scale.requires_grad = self.config.train_logit_scale
            if self.config.train_logit_scale and not self.model_config.pretrained:
                # Initialize logit scale to log(1/0.07) ≈ 2.66 (standard CLIP initialization)
                self.model.logit_scale.data = torch.tensor(np.log(1 / 0.07)).to(self.device)
        
        # Check if there are any trainable parameters
        has_trainable_params = any(p.requires_grad for p in self.model.parameters())
        
        # Check for NaN in model parameters
        has_nan = any(torch.isnan(p).any() for p in self.model.parameters())
        if has_nan:
            print("WARNING: Model contains NaN parameters. Re-initializing trainable parameters.")
            # Re-initialize trainable parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if 'logit_scale' in name:
                        param.data = torch.tensor(np.log(1 / 0.07)).to(self.device)
                    elif 'proj' in name:
                        if param.dim() > 1:
                            torch.nn.init.xavier_uniform_(param)
                        else:
                            torch.nn.init.zeros_(param)
        
        # Test forward pass to check for NaN outputs
        self.model.eval()
        with torch.no_grad():
            test_text = clip.tokenize(["a test sentence"]).to(self.device)
            test_text_features = self.model.encode_text(test_text)
            if torch.isnan(test_text_features).any():
                print("WARNING: Text encoder produces NaN outputs. This may indicate a corrupted model.")
                print("Consider re-downloading the model or checking the model file.")
            else:
                print("Text encoder test: OK")
            
            # Test logit scale
            if hasattr(self.model, 'logit_scale') and self.model.logit_scale is not None:
                logit_scale_val = self.model.logit_scale.exp().item()
                if np.isnan(logit_scale_val) or np.isinf(logit_scale_val):
                    print(f"WARNING: Logit scale is NaN/Inf: {logit_scale_val}. Re-initializing.")
                    self.model.logit_scale.data = torch.tensor(np.log(1 / 0.07)).to(self.device)
                else:
                    print(f"Logit scale: {logit_scale_val:.4f}")
        
        self.model.train()
        
        # Log trainable parameters with dtype and count
        print("\n=== Trainable Parameters ===")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name}: dtype={param.dtype}, shape={tuple(param.shape)}, count={param.numel():,}")
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())
        print(f"Total trainable: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")
        print("=" * 30 + "\n")
        
        # List trainable components
        if hasattr(self.model, 'visual') and any(p.requires_grad for p in self.model.visual.parameters() if not any(p is proj_p for proj_p in image_projection_params)):
            print("  - Vision encoder: trainable")
        if hasattr(self.model, 'transformer') and any(p.requires_grad for p in self.model.transformer.parameters()):
            print("  - Text encoder: trainable")
        if image_projection_params and any(p.requires_grad for p in image_projection_params):
            print("  - Image projection: trainable")
        if hasattr(self.model, 'text_projection') and self.model.text_projection is not None and self.model.text_projection.requires_grad:
            print("  - Text projection: trainable")
        if hasattr(self.model, 'logit_scale') and self.model.logit_scale is not None and self.model.logit_scale.requires_grad:
            print("  - Logit scale: trainable")
        
        # Check model dtype - disable mixed precision if model is already in FP16
        model_dtype = next(self.model.parameters()).dtype
        use_mixed_precision = self.config.mixed_precision and model_dtype == torch.float32
        
        if not use_mixed_precision and self.config.mixed_precision:
            print(f"Model dtype is {model_dtype}, disabling mixed precision")
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer() if has_trainable_params else None
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler() if has_trainable_params else None
        
        # Mixed precision (only if there are trainable parameters and model is FP32)
        self.scaler = None
        if use_mixed_precision and has_trainable_params:
            self.scaler = GradScaler()
            print("Using mixed precision training")
        else:
            print(f"Not using mixed precision (config: {self.config.mixed_precision}, dtype: {model_dtype}, trainable: {has_trainable_params})")
        
        if not has_trainable_params:
            print("WARNING: No trainable parameters found! Model will not be trained.")
            print("Set train_vision_encoder, train_text_encoder, train_image_projection, train_text_projection, or train_logit_scale to true in config.")
        
        # Enable anomaly detection for debugging
        torch.autograd.set_detect_anomaly(True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def _setup_optimizer(self):
        """Setup optimizer with logit_scale excluded from weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'logit_scale' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        print(f"Optimizer: {len(decay_params)} params with weight decay, {len(no_decay_params)} without")
        
        if self.config.optimizer.lower() == "adamw":
            optimizer = AdamW([
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == "adam":
            optimizer = Adam([
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ], lr=self.config.learning_rate)
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
            images = batch["image"].to(self.device)
            texts = clip.tokenize(batch["text"], truncate=True).to(self.device)
            
            # Log first batch completely
            if batch_idx == 0:
                print("\n=== First Batch Diagnostics ===")
                print(f"Captions: {batch['text'][:2]}")
                print(f"Tokens shape: {texts.shape}, dtype: {texts.dtype}")
                print(f"Tokens min/max: {texts.min()}/{texts.max()}")
                if hasattr(self.model, 'logit_scale'):
                    print(f"Logit scale (raw): {self.model.logit_scale.item():.4f}")
                    print(f"Logit scale (exp): {self.model.logit_scale.exp().item():.4f}")
                if hasattr(self.model, 'text_projection'):
                    print(f"Text projection: dtype={self.model.text_projection.dtype}, shape={self.model.text_projection.shape}")
                    print(f"Text projection stats: min={self.model.text_projection.min():.4f}, max={self.model.text_projection.max():.4f}, mean={self.model.text_projection.mean():.4f}")
                print("=" * 30 + "\n")
            
            # Forward pass with mixed precision
            if self.scaler and self.optimizer:
                with autocast():
                    image_features = self.model.encode_image(images)
                    text_features = self.model.encode_text(texts)
                    
                    # Check text features before normalization
                    if batch_idx == 0:
                        print(f"Text features (raw): min={text_features.min():.4f}, max={text_features.max():.4f}, mean={text_features.mean():.4f}")
                        print(f"Text features norm: {text_features.norm(dim=-1).mean():.4f}")
                        if torch.isnan(text_features).any():
                            print("ERROR: Text features contain NaN after encode_text()")
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    loss = self.clip_loss(
                        image_features,
                        text_features,
                        self.model.logit_scale.exp(),
                    )
                
                # Check loss before backward
                if batch_idx == 0:
                    print(f"Loss before backward: {loss.item():.4f}")
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Check gradients after backward
                if batch_idx == 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"ERROR: NaN gradient in {name}")
                                break
                
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
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Check parameters after optimizer step
                    if batch_idx == 0:
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                if torch.isnan(param).any():
                                    print(f"ERROR: NaN parameter after optimizer step in {name}")
                                    break
                        if hasattr(self.model, 'logit_scale'):
                            print(f"Logit scale after step: {self.model.logit_scale.exp().item():.4f}")
            elif self.optimizer:
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # Check text features before normalization
                if batch_idx == 0:
                    print(f"Text features (raw): min={text_features.min():.4f}, max={text_features.max():.4f}, mean={text_features.mean():.4f}")
                    print(f"Text features norm: {text_features.norm(dim=-1).mean():.4f}")
                    if torch.isnan(text_features).any():
                        print("ERROR: Text features contain NaN after encode_text()")
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                loss = self.clip_loss(
                    image_features,
                    text_features,
                    self.model.logit_scale.exp(),
                )
                
                # Check loss before backward
                if batch_idx == 0:
                    print(f"Loss before backward: {loss.item():.4f}")
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Check gradients after backward
                if batch_idx == 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if torch.isnan(param.grad).any():
                                print(f"ERROR: NaN gradient in {name}")
                                break
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip,
                    )
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Check parameters after optimizer step
                    if batch_idx == 0:
                        for name, param in self.model.named_parameters():
                            if param.requires_grad:
                                if torch.isnan(param).any():
                                    print(f"ERROR: NaN parameter after optimizer step in {name}")
                                    break
                        if hasattr(self.model, 'logit_scale'):
                            print(f"Logit scale after step: {self.model.logit_scale.exp().item():.4f}")
            else:
                # No trainable parameters - just compute loss for logging
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # Check text features before normalization
                if batch_idx == 0:
                    print(f"Text features (raw): min={text_features.min():.4f}, max={text_features.max():.4f}, mean={text_features.mean():.4f}")
                    print(f"Text features norm: {text_features.norm(dim=-1).mean():.4f}")
                    if torch.isnan(text_features).any():
                        print("ERROR: Text features contain NaN after encode_text()")
                
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
            self.global_step += 1
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at step {self.global_step}")
                print(f"Image features: min={image_features.min():.4f}, max={image_features.max():.4f}, mean={image_features.mean():.4f}")
                print(f"Text features: min={text_features.min():.4f}, max={text_features.max():.4f}, mean={text_features.mean():.4f}")
                print(f"Logit scale: {self.model.logit_scale.exp().item():.4f}")
                # Skip this batch
                continue
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"] if self.optimizer else 0.0
                
                if self.logger:
                    self.logger.log_metric("train/loss", avg_loss, step=self.global_step)
                    self.logger.log_metric("train/lr", lr, step=self.global_step)
                
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}",
                })
        
        avg_loss = total_loss / num_batches
        
        # Check for NaN average loss
        if np.isnan(avg_loss):
            print(f"WARNING: Average loss is NaN after epoch {self.current_epoch + 1}")
            print(f"Total loss: {total_loss}, Num batches: {num_batches}")
            avg_loss = 0.0  # Set to 0 to avoid NaN in logs
        
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
            images = batch["image"].to(self.device)
            texts = clip.tokenize(batch["text"], truncate=True).to(self.device)
            
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
