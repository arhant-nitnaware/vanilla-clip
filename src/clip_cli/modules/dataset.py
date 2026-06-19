"""
Dataset module for CLIP training and evaluation
Supports custom datasets with image-text pairs
Supports multiple captions per image
"""

import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import random


class CLIPDataset(Dataset):
    """Generic image-text dataset for CLIP with multi-caption support"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
        sample_caption: bool = True,
    ):
        """
        Initialize CLIP dataset
        
        Args:
            data_path: Path to dataset directory
            split: Dataset split (train, val, test)
            transform: Optional transform to apply to images
            image_size: Size to resize images to
            sample_caption: If True, randomly sample one caption per image during training.
                          If False, return all captions (for evaluation).
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.sample_caption = sample_caption
        
        # Default transform if none provided
        if transform is None:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Group captions by image
        self.image_to_captions = self._group_captions_by_image()
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load image-text pairs from CSV or directory structure"""
        # Try to load from CSV
        csv_path = self.data_path / f"{self.split}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df.to_dict('records')
        
        # Try to load from directory structure
        images_dir = self.data_path / self.split / "images"
        if images_dir.exists():
            data = []
            for image_path in images_dir.glob("*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Use filename as text (can be overridden)
                    text = image_path.stem.replace('_', ' ')
                    data.append({
                        "image": str(image_path.relative_to(self.data_path)),
                        "text": text,
                    })
            return data
        
        raise FileNotFoundError(
            f"No data found for split '{self.split}'. "
            f"Expected {csv_path} or {images_dir}"
        )
    
    def _group_captions_by_image(self) -> Dict[str, List[str]]:
        """Group all captions by image path"""
        image_captions = {}
        for item in self.data:
            image_path = item["image"]
            text = item["text"]
            
            if image_path not in image_captions:
                image_captions[image_path] = []
            image_captions[image_path].append(text)
        
        return image_captions
    
    def __len__(self) -> int:
        """Return number of unique images"""
        return len(self.image_to_captions)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a single sample
        
        Returns:
            Dictionary with 'image', 'text', 'image_path', and optionally 'all_texts'
        """
        # Get image path and all captions
        image_path = list(self.image_to_captions.keys())[idx]
        all_captions = self.image_to_captions[image_path]
        
        # Sample one caption if in training mode
        if self.sample_caption:
            text = random.choice(all_captions)
        else:
            # Return first caption for compatibility, but also provide all
            text = all_captions[0]
        
        # Load image
        full_image_path = self.data_path / image_path
        image = Image.open(full_image_path).convert("RGB")
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        result = {
            "image": image,
            "text": text,
            "image_path": str(full_image_path),
        }
        
        # Include all captions if not sampling (for evaluation)
        if not self.sample_caption:
            result["all_texts"] = all_captions
        
        return result


def create_dataloader(
    dataset: CLIPDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for CLIP dataset
    
    Args:
        dataset: CLIPDataset instance
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_train_val_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    transform: Optional[Callable] = None,
    sample_caption_train: bool = True,
    sample_caption_val: bool = False,
) -> tuple:
    """
    Create training and validation dataloaders
    
    Args:
        data_path: Path to dataset directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Size to resize images to
        transform: Optional transform to apply to images
        sample_caption_train: Whether to sample random caption during training
        sample_caption_val: Whether to sample random caption during validation
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = CLIPDataset(
        data_path=data_path,
        split="train",
        transform=transform,
        image_size=image_size,
        sample_caption=sample_caption_train,
    )
    
    val_dataset = CLIPDataset(
        data_path=data_path,
        split="val",
        transform=transform,
        image_size=image_size,
        sample_caption=sample_caption_val,
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    
    return train_loader, val_loader
