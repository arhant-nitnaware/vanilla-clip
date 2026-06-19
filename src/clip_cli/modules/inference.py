"""
inference module for CLIP
Supports image-text similarity, zero-shot classification, and retrieval
"""

import clip
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from ..core.logger import ExperimentLogger


class CLIPInference:
    """CLIP inference engine with multiple capabilities"""
    
    def __init__(
        self,
        model,
        preprocess,
        logger: Optional[ExperimentLogger] = None,
        device: str = "auto",
    ):
        self.model = model
        self.preprocess = preprocess
        self.logger = logger
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def _get_device(self, device: str) -> torch.device:
        """Resolve device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @torch.no_grad()
    def encode_image(
        self,
        image_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Encode an image into a feature vector
        
        Args:
            image_path: Path to image file
        
        Returns:
            Normalized image feature vector [1, D]
        """
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        features = self.model.encode_image(image)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:
        """
        Encode text into a feature vector
        
        Args:
            text: Text string
        
        Returns:
            Normalized text feature vector [1, D]
        """
        tokens = clip.tokenize([text], truncate=True).to(self.device)
        
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_images_batch(
        self,
        image_paths: List[Union[str, Path]],
    ) -> torch.Tensor:
        """
        Encode multiple images into feature vectors
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            Normalized image feature vectors [N, D]
        """
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            images.append(self.preprocess(image))
        
        images = torch.stack(images).to(self.device)
        
        features = self.model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def encode_texts_batch(
        self,
        texts: List[str],
    ) -> torch.Tensor:
        """
        Encode multiple texts into feature vectors
        
        Args:
            texts: List of text strings
        
        Returns:
            Normalized text feature vectors [N, D]
        """
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.cpu()
    
    @torch.no_grad()
    def compute_similarity(
        self,
        image_path: Union[str, Path],
        text: str,
    ) -> float:
        """
        Compute cosine similarity between an image and text
        
        Args:
            image_path: Path to image file
            text: Text string
        
        Returns:
            Similarity score (0-1)
        """
        image_features = self.encode_image(image_path)
        text_features = self.encode_text(text)
        
        similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    @torch.no_grad()
    def zero_shot_classify(
        self,
        image_path: Union[str, Path],
        labels: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, float]]:
        """
        Perform zero-shot classification on an image
        
        Args:
            image_path: Path to image file
            labels: List of class labels
            top_k: Number of top results to return
        
        Returns:
            List of dictionaries with 'label' and 'score' keys
        """
        # Encode image
        image_features = self.encode_image(image_path)
        
        # Encode all labels
        text_features = self.encode_texts_batch(labels)
        
        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        
        # Get top-k
        top_k_values, top_k_indices = similarities.topk(min(top_k, len(labels)))
        
        results = []
        for score, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
            results.append({
                "label": labels[idx],
                "score": score,
            })
        
        return results
    
    @torch.no_grad()
    def retrieve_images(
        self,
        query_text: str,
        image_paths: List[Union[str, Path]],
        top_k: int = 5,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve most similar images for a text query
        
        Args:
            query_text: Text query
            image_paths: List of image paths to search
            top_k: Number of top results to return
        
        Returns:
            List of dictionaries with 'image_path' and 'score' keys
        """
        # Encode query text
        query_features = self.encode_text(query_text)
        
        # Encode all images
        image_features = self.encode_images_batch(image_paths)
        
        # Compute similarities
        similarities = (image_features @ query_features.T).squeeze(1)
        
        # Get top-k
        top_k_values, top_k_indices = similarities.topk(min(top_k, len(image_paths)))
        
        results = []
        for score, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
            results.append({
                "image_path": str(image_paths[idx]),
                "score": score,
            })
        
        return results
    
    @torch.no_grad()
    def retrieve_texts(
        self,
        query_image: Union[str, Path],
        texts: List[str],
        top_k: int = 5,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve most similar texts for an image query
        
        Args:
            query_image: Path to query image
            texts: List of texts to search
            top_k: Number of top results to return
        
        Returns:
            List of dictionaries with 'text' and 'score' keys
        """
        # Encode query image
        query_features = self.encode_image(query_image)
        
        # Encode all texts
        text_features = self.encode_texts_batch(texts)
        
        # Compute similarities
        similarities = (query_features @ text_features.T).squeeze(0)
        
        # Get top-k
        top_k_values, top_k_indices = similarities.topk(min(top_k, len(texts)))
        
        results = []
        for score, idx in zip(top_k_values.tolist(), top_k_indices.tolist()):
            results.append({
                "text": texts[idx],
                "score": score,
            })
        
        return results
    
    @torch.no_grad()
    def batch_similarity(
        self,
        image_paths: List[Union[str, Path]],
        texts: List[str],
    ) -> np.ndarray:
        """
        Compute pairwise similarities between images and texts
        
        Args:
            image_paths: List of image paths
            texts: List of texts
        
        Returns:
            Similarity matrix [len(image_paths), len(texts)]
        """
        # Encode images
        image_features = self.encode_images_batch(image_paths)
        
        # Encode texts
        text_features = self.encode_texts_batch(texts)
        
        # Compute similarity matrix
        similarity_matrix = (image_features @ text_features.T).numpy()
        
        return similarity_matrix
    
    def save_embedding(
        self,
        features: torch.Tensor,
        save_path: Union[str, Path],
    ):
        """
        Save embedding to file
        
        Args:
            features: Feature tensor
            save_path: Path to save embedding
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, features.numpy())
        
        if self.logger:
            self.logger.info(f"Embedding saved to {save_path}")
    
    def load_embedding(
        self,
        load_path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Load embedding from file
        
        Args:
            load_path: Path to load embedding from
        
        Returns:
            Feature tensor
        """
        features = np.load(load_path)
        return torch.from_numpy(features)
