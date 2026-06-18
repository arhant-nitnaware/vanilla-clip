"""
Offline-first model loading with caching and verification
Supports local cache with online fallback
"""

import os
import shutil
import urllib.request
import urllib.error
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

import torch


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    url: str
    checksum: str
    size_mb: float


class ModelLoader:
    """Offline-first model loader with caching"""
    
    # Official OpenAI CLIP model URLs
    MODEL_REGISTRY: Dict[str, ModelInfo] = {
        "RN50": ModelInfo(
            name="RN50",
            url="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
            checksum="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
            size_mb=102.4,
        ),
        "RN101": ModelInfo(
            name="RN101",
            url="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
            checksum="8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599",
            size_mb=174.1,
        ),
        "RN50x4": ModelInfo(
            name="RN50x4",
            url="https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
            checksum="7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd",
            size_mb=345.3,
        ),
        "RN50x16": ModelInfo(
            name="RN50x16",
            url="https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
            checksum="52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa",
            size_mb=1238.6,
        ),
        "RN50x64": ModelInfo(
            name="RN50x64",
            url="https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
            checksum="be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c",
            size_mb=4705.2,
        ),
        "ViT-B-32": ModelInfo(
            name="ViT-B-32",
            url="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
            checksum="40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
            size_mb=338.0,
        ),
        "ViT-B-16": ModelInfo(
            name="ViT-B-16",
            url="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
            checksum="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f",
            size_mb=568.0,
        ),
        "ViT-L-14": ModelInfo(
            name="ViT-L-14",
            url="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
            checksum="b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
            size_mb=755.5,
        ),
        "ViT-L-14-336px": ModelInfo(
            name="ViT-L-14-336px",
            url="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
            checksum="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
            size_mb=1355.0,
        ),
    }
    
    def __init__(
        self,
        cache_dir: str = "cache/models",
        download_dir: str = "cache/downloads",
        offline_mode: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.offline_mode = offline_mode
        
        # Set environment variables for third-party libraries
        self._setup_cache_dirs()
    
    def _setup_cache_dirs(self):
        """Setup cache directories for third-party libraries"""
        base_cache = self.cache_dir.parent
        os.environ["TORCH_HOME"] = str(base_cache / "torch")
        os.environ["HF_HOME"] = str(base_cache / "huggingface")
        os.environ["TRANSFORMERS_CACHE"] = str(base_cache / "huggingface")
        os.environ["XDG_CACHE_HOME"] = str(base_cache)
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get local cache path for a model"""
        return self.cache_dir / f"{model_name}.pt"
    
    def _model_exists(self, model_name: str) -> bool:
        """Check if model exists in local cache"""
        return self._get_model_path(model_name).exists()
    
    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        actual_checksum = self._compute_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def _download_model(
        self,
        model_name: str,
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> Path:
        """Download model with retry logic and progress"""
        if self.offline_mode:
            raise RuntimeError(
                f"Offline mode enabled: Cannot download {model_name}. "
                "Please download the model manually or disable offline mode."
            )
        
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(self.MODEL_REGISTRY.keys())}"
            )
        
        model_info = self.MODEL_REGISTRY[model_name]
        destination = self._get_model_path(model_name)
        
        if destination.exists():
            if self._verify_checksum(destination, model_info.checksum):
                print(f"Model {model_name} already cached and verified")
                return destination
            else:
                print(f"Model {model_name} cache corrupted, re-downloading...")
                destination.unlink()
        
        url = model_info.url
        
        for attempt in range(max_retries):
            try:
                print(f"Downloading {model_name} ({model_info.size_mb:.1f} MB)...")
                print(f"Attempt {attempt + 1}/{max_retries}")
                
                with urllib.request.urlopen(url) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    downloaded = 0
                    
                    with open(destination, "wb") as output:
                        while True:
                            chunk = response.read(block_size)
                            if not chunk:
                                break
                            
                            output.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                print(f"\rProgress: {progress:.1f}%", end="", flush=True)
                
                print(f"\nDownloaded {model_name}")
                
                # Verify checksum
                if not self._verify_checksum(destination, model_info.checksum):
                    destination.unlink()
                    raise RuntimeError(
                        f"Checksum verification failed for {model_name}. "
                        f"Downloaded file may be corrupted."
                    )
                
                print(f"Model {model_name} verified and cached")
                return destination
                
            except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError) as e:
                if destination.exists():
                    destination.unlink()
                
                if attempt < max_retries - 1:
                    print(f"\nDownload failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to download {model_name} after {max_retries} attempts: {e}"
                    ) from e
    
    def load_model(
        self,
        model_name: str = "ViT-B-32",
        device: str = "auto",
        jit: bool = False,
        download_if_missing: bool = True,
    ):
        """
        Load CLIP model with offline-first behavior
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on ('auto', 'cuda', 'cpu')
            jit: Whether to load JIT compiled model
            download_if_missing: Whether to download model if not in cache
        
        Returns:
            model, preprocess
        """
        import clip
        
        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model exists locally
        if self._model_exists(model_name):
            model_path = self._get_model_path(model_name)
            print(f"Loading {model_name} from local cache: {model_path}")
        elif download_if_missing and not self.offline_mode:
            model_path = self._download_model(model_name)
        else:
            raise RuntimeError(
                f"Model {model_name} not found in cache and offline mode enabled. "
                "Please download the model manually or disable offline mode."
            )
        
        # Load model
        model, preprocess = clip.load(str(model_path), device=device, jit=jit)
        
        return model, preprocess
    
    def list_cached_models(self) -> list:
        """List all cached models"""
        cached = []
        for model_file in self.cache_dir.glob("*.pt"):
            model_name = model_file.stem
            if model_name in self.MODEL_REGISTRY:
                cached.append({
                    "name": model_name,
                    "path": str(model_file),
                    "size_mb": model_file.stat().st_size / (1024 * 1024),
                })
        return cached
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models"""
        if model_name:
            model_path = self._get_model_path(model_name)
            if model_path.exists():
                model_path.unlink()
                print(f"Cleared cache for {model_name}")
        else:
            for model_file in self.cache_dir.glob("*.pt"):
                model_file.unlink()
            print("Cleared all model cache")
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        return self.MODEL_REGISTRY.get(model_name)
    
    def list_available_models(self) -> list:
        """List all available models"""
        return list(self.MODEL_REGISTRY.keys())
