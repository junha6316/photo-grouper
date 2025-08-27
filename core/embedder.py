import torch
from PIL import Image
import numpy as np
from typing import Dict, Optional, Tuple
import hashlib
from sklearn.decomposition import PCA
from infra.cache_db import EmbeddingCache
from .models import BaseModel, VGG16Model, ResNet18Model

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
    def __init__(self, model_type: str = "resnet18", device: Optional[str] = None, pca_components: int = 512):
        """
        Initialize the embedder with specified model type.
        
        Args:
            model_type: Type of model to use ("vgg16", "resnet18")
            device: PyTorch device ('cuda', 'cpu', or None for auto)
            pca_components: Number of PCA components
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pca_components = pca_components
        self.pca = None
        self.pca_fitted = False
        
        # Initialize model
        self.model_impl = self._get_model_impl(model_type)
        self.feature_dim = self.model_impl.get_feature_dim()
        self.model_name = self.model_impl.get_model_name()
        
        self._load_model()
        self._setup_transforms()
        
        # Initialize database cache
        self.db_cache = EmbeddingCache()
        # In-memory cache for current session
        self._embedding_cache = {}
        self._raw_embeddings_cache = {}
    
    def _get_model_impl(self, model_type: str) -> BaseModel:
        """Get model implementation based on type."""
        if model_type == "vgg16":
            return VGG16Model()
        elif model_type == "resnet18":
            return ResNet18Model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_model(self):
        """Load the specified model."""
        self.model = self.model_impl.get_model()
        self.model.eval()
        self.model.to(self.device)
        print(f"Loaded {self.model_name} model with efficient global average pooling ({self.feature_dim} features)")
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transform = self.model_impl.get_transforms()

    def _get_raw_embedding(self, image_path: str) -> np.ndarray:
        """Extract raw model embedding without PCA."""
        # Check in-memory cache first
        image_hash = self._get_image_hash(image_path)
        if image_hash in self._raw_embeddings_cache:
            return self._raw_embeddings_cache[image_hash]
        
        # Check database cache
        cached_embedding = self.db_cache.get_embedding(
            image_path, f"{self.model_name}_raw"
        )
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
            self.db_cache.save_embedding(
                image_path, embedding, f"{self.model_name}_raw"
            )
            return embedding
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return zero vector for failed images with correct dimensions
            return np.zeros(self.feature_dim, dtype=np.float32)

    def _get_raw_embeddings_batch(self, image_paths: list, batch_size: int = 32) -> list:
        """Process multiple images in batches for better GPU utilization."""
        all_embeddings = []
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Load and preprocess batch
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    batch_tensors.append(torch.zeros(3, 224, 224))
            
            # Stack into single batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Process batch
            with torch.no_grad():
                features = self.model(batch_tensor)
                # Normalize each feature vector
                features = torch.nn.functional.normalize(features, p=2, dim=1)
                batch_embeddings = features.cpu().numpy()
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def fit_pca(self, image_paths: list, batch_size: int = 32):
        """Fit PCA on a set of images using batch processing."""
        print("Fitting PCA on image embeddings...")
        
        # Process images in batches
        raw_embeddings = self._get_raw_embeddings_batch(image_paths, batch_size)
        
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
            cached_pca_embedding = self.db_cache.get_embedding(
                image_path, f"{self.model_name}_pca"
            )
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
            self.db_cache.save_embedding(image_path, pca_embedding, f"{self.model_name}_pca")
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
        if model_name is None or model_name == f"{self.model_name}_raw":
            self._raw_embeddings_cache.clear()
        if model_name is None or model_name == f"{self.model_name}_pca":
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