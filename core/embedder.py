import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple
import hashlib
from sklearn.decomposition import PCA
from infra.cache_db import EmbeddingCache

# Initialize HEIF support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_AVAILABLE = True
    print("HEIC/HEIF support enabled")
except ImportError:
    HEIF_AVAILABLE = False
    print("HEIC/HEIF support not available (pillow-heif not installed)")


class ImageEmbedder:
    def __init__(self, device: Optional[str] = None, pca_components: int = 512):
        """
        Initialize the embedder with VGG16 model using efficient global average pooling.
        This implementation matches the reference code approach with 512 features directly.
        
        Args:
            device: PyTorch device ('cuda', 'cpu', or None for auto)
            pca_components: Number of PCA components (default: 512, same as VGG16 features)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pca_components = pca_components
        self.pca = None
        self.pca_fitted = False
        self.feature_dim = 512  # VGG16 with global average pooling gives 512 features
        
        self._load_model()
        self._setup_transforms()
        
        # Initialize database cache
        self.db_cache = EmbeddingCache()
        # In-memory cache for current session
        self._embedding_cache = {}
        self._raw_embeddings_cache = {}
    
    def _load_model(self):
        """Load VGG16 with efficient global average pooling (like reference code)."""
        # Load VGG16 base model
        base_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # VGG16: features + global average pooling (matches reference implementation)
        self.model = nn.Sequential(
            base_model.features,
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling -> 512 features
            nn.Flatten()
        )
        
        self.model.eval()
        self.model.to(self.device)
        print(f"Loaded VGG16 model with efficient global average pooling ({self.feature_dim} features)")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_raw_embedding(self, image_path: str) -> np.ndarray:
        """Extract raw model embedding without PCA."""
        # Check in-memory cache first
        image_hash = self._get_image_hash(image_path)
        if image_hash in self._raw_embeddings_cache:
            return self._raw_embeddings_cache[image_hash]
        
        # Check database cache
        cached_embedding = self.db_cache.get_embedding(image_path, "vgg16_raw")
        if cached_embedding is not None:
            self._raw_embeddings_cache[image_hash] = cached_embedding
            return cached_embedding
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
                # Features are already flattened by our model
                features = features.squeeze()
                # L2 normalize
                features = torch.nn.functional.normalize(features, p=2, dim=0)
                embedding = features.cpu().numpy()
            
            # Cache the result in both memory and database
            self._raw_embeddings_cache[image_hash] = embedding
            self.db_cache.save_embedding(image_path, embedding, "vgg16_raw")
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return zero vector for failed images with correct dimensions
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def fit_pca(self, image_paths: list):
        """Fit PCA on a set of images."""
        print("Fitting PCA on image embeddings...")
        raw_embeddings = []
        
        for i, path in enumerate(image_paths):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(image_paths)} for PCA fitting...")
            raw_embedding = self._get_raw_embedding(path)
            raw_embeddings.append(raw_embedding)
        
        # Fit PCA
        raw_embeddings_matrix = np.array(raw_embeddings)
        self.pca = PCA(n_components=self.pca_components)
        self.pca.fit(raw_embeddings_matrix)
        self.pca_fitted = True
        
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA fitted with {self.pca_components} components, "
              f"explaining {explained_variance:.3f} of the variance")
    
    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract PCA-reduced embedding for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized PCA-reduced embedding vector
        """
        # Check in-memory cache first
        image_hash = self._get_image_hash(image_path)
        if image_hash in self._embedding_cache:
            return self._embedding_cache[image_hash]
        
        # If PCA is fitted, check database cache for PCA embedding
        if self.pca_fitted and self.pca is not None:
            cached_pca_embedding = self.db_cache.get_embedding(image_path, "vgg16_pca")
            if cached_pca_embedding is not None:
                self._embedding_cache[image_hash] = cached_pca_embedding
                return cached_pca_embedding
        
        # Get raw embedding
        raw_embedding = self._get_raw_embedding(image_path)
        
        # Apply PCA if fitted
        if self.pca_fitted and self.pca is not None:
            pca_embedding = self.pca.transform([raw_embedding])[0]
            # L2 normalize the PCA result
            pca_embedding = pca_embedding / (np.linalg.norm(pca_embedding) + 1e-8)
            
            # Cache PCA embedding in both memory and database
            self._embedding_cache[image_hash] = pca_embedding
            self.db_cache.save_embedding(image_path, pca_embedding, "vgg16_pca")
            return pca_embedding
        else:
            # If PCA not fitted, return raw embedding
            self._embedding_cache[image_hash] = raw_embedding
            return raw_embedding
    
    def get_embeddings_batch(self, image_paths: list) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        for path in image_paths:
            embeddings[path] = self.get_embedding(path)
        return embeddings
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.db_cache.get_cache_stats()
        stats['memory_cache_raw'] = len(self._raw_embeddings_cache)
        stats['memory_cache_pca'] = len(self._embedding_cache)
        return stats
    
    def clear_cache(self, model_name: str = None) -> int:
        """Clear cache entries."""
        # Clear memory caches
        if model_name is None or model_name == "vgg16_raw":
            self._raw_embeddings_cache.clear()
        if model_name is None or model_name == "vgg16_pca":
            self._embedding_cache.clear()
        
        # Clear database cache
        return self.db_cache.clear_cache(model_name)
    
    def _get_image_hash(self, image_path: str) -> str:
        """Generate a hash for caching based on file path and modification time."""
        import os
        try:
            stat = os.stat(image_path)
            hash_string = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_string.encode()).hexdigest()
        except:
            return hashlib.md5(image_path.encode()).hexdigest()