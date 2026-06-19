"""
evaluation module for CLIP
Supports comprehensive metrics including recall, precision, MRR, and mean rank
"""

import clip
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..core.logger import ExperimentLogger


class CLIPEvaluator:
    """Comprehensive CLIP evaluator with multiple metrics"""
    
    def __init__(
        self,
        model,
        val_loader: DataLoader,
        logger: Optional[ExperimentLogger] = None,
        device: str = "auto",
    ):
        self.model = model
        self.val_loader = val_loader
        self.logger = logger
        self.device = self._get_device(device)
        self.model.to(self.device)
    
    def _get_device(self, device: str) -> torch.device:
        """Resolve device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @torch.no_grad()
    def extract_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract image and text embeddings from validation set
        
        Returns:
            image_features: [N, D] image embeddings
            text_features: [N, D] text embeddings
        """
        self.model.eval()
        
        image_features_list = []
        text_features_list = []
        
        progress_bar = tqdm(self.val_loader, desc="Extracting embeddings")
        
        for batch in progress_bar:
            images = batch["images"].to(self.device)
            texts = clip.tokenize(batch["texts"], truncate=True).to(self.device)
            
            # Extract features
            img_feat = self.model.encode_image(images)
            txt_feat = self.model.encode_text(texts)
            
            # Normalize
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            
            image_features_list.append(img_feat.cpu())
            text_features_list.append(txt_feat.cpu())
        
        # Concatenate all features
        image_features = torch.cat(image_features_list, dim=0)
        text_features = torch.cat(text_features_list, dim=0)
        
        return image_features, text_features
    
    def compute_similarity_matrix(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity matrix
        
        Args:
            image_features: [N, D] image embeddings
            text_features: [N, D] text embeddings
        
        Returns:
            similarity_matrix: [N, N] similarity scores
        """
        return image_features @ text_features.T
    
    def compute_recall_at_k(
        self,
        similarity_matrix: torch.Tensor,
        k: int = 1,
    ) -> float:
        """
        Compute Recall@K
        
        Args:
            similarity_matrix: [N, N] similarity scores
            k: Top-k to consider
        
        Returns:
            recall@k score
        """
        n = similarity_matrix.shape[0]
        targets = torch.arange(n, device=similarity_matrix.device)
        
        # Get top-k indices
        top_k_indices = similarity_matrix.topk(k, dim=1).indices
        
        # Check if target is in top-k
        correct = (top_k_indices == targets.unsqueeze(1)).any(dim=1)
        
        return correct.float().mean().item()
    
    def compute_precision_at_k(
        self,
        similarity_matrix: torch.Tensor,
        k: int = 1,
    ) -> float:
        """
        Compute Precision@K
        
        Args:
            similarity_matrix: [N, N] similarity scores
            k: Top-k to consider
        
        Returns:
            precision@k score
        """
        n = similarity_matrix.shape[0]
        targets = torch.arange(n, device=similarity_matrix.device)
        
        # Get top-k indices
        top_k_indices = similarity_matrix.topk(k, dim=1).indices
        
        # Check if target is in top-k
        correct = (top_k_indices == targets.unsqueeze(1)).float()
        
        # Precision = correct / k
        precision = correct.sum(dim=1) / k
        
        return precision.mean().item()
    
    def compute_mean_reciprocal_rank(
        self,
        similarity_matrix: torch.Tensor,
    ) -> float:
        """
        Compute Mean Reciprocal Rank (MRR)
        
        Args:
            similarity_matrix: [N, N] similarity scores
        
        Returns:
            MRR score
        """
        n = similarity_matrix.shape[0]
        targets = torch.arange(n, device=similarity_matrix.device)
        
        # Get rankings (sorted by similarity, descending)
        rankings = similarity_matrix.argsort(dim=1, descending=True)
        
        reciprocal_ranks = []
        
        for i in range(n):
            # Find rank of correct match
            rank = (rankings[i] == targets[i]).nonzero(as_tuple=True)[0].item()
            reciprocal_ranks.append(1.0 / (rank + 1))
        
        return np.mean(reciprocal_ranks)
    
    def compute_mean_rank(
        self,
        similarity_matrix: torch.Tensor,
    ) -> float:
        """
        Compute Mean Rank (lower is better)
        
        Args:
            similarity_matrix: [N, N] similarity scores
        
        Returns:
            Mean rank
        """
        n = similarity_matrix.shape[0]
        targets = torch.arange(n, device=similarity_matrix.device)
        
        # Get rankings (sorted by similarity, descending)
        rankings = similarity_matrix.argsort(dim=1, descending=True)
        
        ranks = []
        
        for i in range(n):
            # Find rank of correct match
            rank = (rankings[i] == targets[i]).nonzero(as_tuple=True)[0].item()
            ranks.append(rank + 1)
        
        return np.mean(ranks)
    
    def compute_median_rank(
        self,
        similarity_matrix: torch.Tensor,
    ) -> float:
        """
        Compute Median Rank (lower is better)
        
        Args:
            similarity_matrix: [N, N] similarity scores
        
        Returns:
            Median rank
        """
        n = similarity_matrix.shape[0]
        targets = torch.arange(n, device=similarity_matrix.device)
        
        # Get rankings (sorted by similarity, descending)
        rankings = similarity_matrix.argsort(dim=1, descending=True)
        
        ranks = []
        
        for i in range(n):
            # Find rank of correct match
            rank = (rankings[i] == targets[i]).nonzero(as_tuple=True)[0].item()
            ranks.append(rank + 1)
        
        return np.median(ranks)
    
    def evaluate(
        self,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation
        
        Args:
            metrics: List of metrics to compute. If None, computes all.
        
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = [
                "recall@1", "recall@5", "recall@10",
                "precision@1", "precision@5",
                "mrr", "mean_rank", "median_rank",
            ]
        
        if self.logger:
            self.logger.info("Starting evaluation...")
        
        # Extract embeddings
        image_features, text_features = self.extract_embeddings()
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        # Compute metrics
        results = {}
        
        for metric in metrics:
            if metric.startswith("recall@"):
                k = int(metric.split("@")[1])
                results[metric] = self.compute_recall_at_k(similarity_matrix, k)
            elif metric.startswith("precision@"):
                k = int(metric.split("@")[1])
                results[metric] = self.compute_precision_at_k(similarity_matrix, k)
            elif metric == "mrr":
                results[metric] = self.compute_mean_reciprocal_rank(similarity_matrix)
            elif metric == "mean_rank":
                results[metric] = self.compute_mean_rank(similarity_matrix)
            elif metric == "median_rank":
                results[metric] = self.compute_median_rank(similarity_matrix)
            else:
                if self.logger:
                    self.logger.warning(f"Unknown metric: {metric}")
        
        # Log results
        if self.logger:
            self.logger.info("Evaluation results:")
            for metric, value in results.items():
                self.logger.info(f"  {metric}: {value:.4f}")
            self.logger.log_metrics(results)
        
        return results
    
    def evaluate_retrieval(
        self,
        query_type: str = "image_to_text",
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance
        
        Args:
            query_type: "image_to_text" or "text_to_image"
            k: Top-k for retrieval
        
        Returns:
            Dictionary of retrieval metrics
        """
        if self.logger:
            self.logger.info(f"Evaluating {query_type} retrieval...")
        
        # Extract embeddings
        image_features, text_features = self.extract_embeddings()
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(image_features, text_features)
        
        if query_type == "text_to_image":
            similarity_matrix = similarity_matrix.T
        
        # Compute retrieval metrics
        results = {
            f"recall@{k}": self.compute_recall_at_k(similarity_matrix, k),
            f"precision@{k}": self.compute_precision_at_k(similarity_matrix, k),
            "mrr": self.compute_mean_reciprocal_rank(similarity_matrix),
            "mean_rank": self.compute_mean_rank(similarity_matrix),
        }
        
        if self.logger:
            self.logger.info(f"{query_type} retrieval results:")
            for metric, value in results.items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
