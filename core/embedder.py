import torch
from PIL import Image
import numpy as np
from typing import Dict, Optional
import hashlib
from sklearn.decomposition import PCA
from infra.cache_db import EmbeddingCache
from .models import BaseModel, VGG16Model, ResNet18Model, MobileNetV3SmallModel

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
    def __init__(self, model_type: str = "mobilenetv3_small", device: Optional[str] = None, pca_components: int = 512):
        """
        Initialize the embedder with specified model type.
        
        Args:
            model_type: Type of model to use ("vgg16", "resnet18", "mobilenetv3_small")
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
        
        # Try to load PCA model from cache on initialization
        self._load_cached_pca_model()
    
    def _get_model_impl(self, model_type: str) -> BaseModel:
        """Get model implementation based on type."""
        if model_type == "vgg16":
            return VGG16Model()
        elif model_type == "resnet18":
            return ResNet18Model()
        elif model_type == "mobilenetv3_small":
            print("Using MobileNetV3-Small model")
            return MobileNetV3SmallModel()
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
    
    def _load_cached_pca_model(self):
        """Try to load PCA model from cache on initialization."""
        try:
            cached_pca = self.db_cache.load_pca_model(self.model_name, self.pca_components)
            if cached_pca is not None:
                self.pca = cached_pca
                self.pca_fitted = True
        except Exception:
            # If loading fails, just continue without cached PCA
            pass

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

    def fit_pca(self, image_paths: list, batch_size: int = 32, force_refit: bool = False):
        """
        Fit PCA on a set of images using batch processing.
        First tries to load from cache, then fits if needed.
        """
        # Try to load PCA model from cache first (unless forced to refit)
        if not force_refit:
            cached_pca = self.db_cache.load_pca_model(self.model_name, self.pca_components)
            if cached_pca is not None:
                self.pca = cached_pca
                self.pca_fitted = True
                return
        
        print("Fitting PCA on image embeddings...")
        
        # Process images in batches
        raw_embeddings = self._get_raw_embeddings_batch(image_paths, batch_size)
        
        # Fit PCA
        raw_embeddings_matrix = np.array(raw_embeddings)
        self.pca = PCA(n_components=self.pca_components)
        self.pca.fit(raw_embeddings_matrix)
        self.pca_fitted = True
        
        # Save PCA model to cache
        self.db_cache.save_pca_model(self.pca, self.model_name, self.pca_components)
        
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

    def get_embeddings_batch(self, image_paths: list, batch_size: int = 32, progress_callback=None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for multiple images using efficient batch processing.
        
        Args:
            image_paths: List of image file paths
            batch_size: Number of images to process in each batch
            progress_callback: Optional callback function for progress updates (index, total, path)
            
        Returns:
            Dictionary mapping image paths to embeddings
        """
        embeddings = {}
        total_images = len(image_paths)
        
        # Process images in batches for better GPU utilization
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]
            
            # Check cache first for this batch
            batch_uncached = []
            batch_uncached_indices = []
            
            for i, path in enumerate(batch_paths):
                image_hash = self._get_image_hash(path)
                
                # Check memory cache first
                if image_hash in self._embedding_cache:
                    embeddings[path] = self._embedding_cache[image_hash]
                    if progress_callback:
                        progress_callback(batch_start + i, total_images, path)
                    continue
                
                # Check database cache for PCA embedding
                if self.pca_fitted and self.pca is not None:
                    cached_pca_embedding = self.db_cache.get_embedding(
                        path, f"{self.model_name}_pca"
                    )
                    if cached_pca_embedding is not None:
                        self._embedding_cache[image_hash] = cached_pca_embedding
                        embeddings[path] = cached_pca_embedding
                        if progress_callback:
                            progress_callback(batch_start + i, total_images, path)
                        continue
                
                # Need to compute embedding
                batch_uncached.append(path)
                batch_uncached_indices.append(batch_start + i)
            
            # Process uncached images in batch if any
            if batch_uncached:
                self._process_uncached_batch(
                    batch_uncached, batch_uncached_indices,
                    embeddings, progress_callback, total_images)
        
        return embeddings
    
    def _process_uncached_batch(self, batch_paths: list, batch_indices: list,
                                embeddings: dict, progress_callback,
                                total_images: int):
        """Process a batch of uncached images efficiently."""
        batch_tensors = []
        valid_paths = []
        
        # Load and preprocess batch
        for path in batch_paths:
            try:
                image = Image.open(path).convert('RGB')
                tensor = self.transform(image)
                batch_tensors.append(tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                # Use zero embedding for failed images
                dim = self.pca_components if self.pca_fitted else self.feature_dim
                embeddings[path] = np.zeros(dim, dtype=np.float32)
        
        if not batch_tensors:
            return
        
        # Stack into single batch tensor and process
        batch_tensor = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            # Extract raw features
            features = self.model(batch_tensor)
            # Normalize each feature vector
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            raw_embeddings = features.cpu().numpy()
        
        # Apply PCA and cache results
        for i, path in enumerate(valid_paths):
            raw_embedding = raw_embeddings[i]
            image_hash = self._get_image_hash(path)
            
            # Cache raw embedding
            self._raw_embeddings_cache[image_hash] = raw_embedding
            self.db_cache.save_embedding(path, raw_embedding, f"{self.model_name}_raw")
            
            # Apply PCA if fitted
            if self.pca_fitted and self.pca is not None:
                pca_embedding = self.pca.transform([raw_embedding])[0]
                # L2 normalize the PCA result
                pca_embedding = pca_embedding / (np.linalg.norm(pca_embedding) + 1e-8)
                
                # Cache PCA embedding
                self._embedding_cache[image_hash] = pca_embedding
                self.db_cache.save_embedding(path, pca_embedding, f"{self.model_name}_pca")
                embeddings[path] = pca_embedding
            else:
                self._embedding_cache[image_hash] = raw_embedding
                embeddings[path] = raw_embedding
            
            # Update progress
            if progress_callback:
                batch_idx = batch_indices[valid_paths.index(path)]
                progress_callback(batch_idx, total_images, path)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        stats = self.db_cache.get_cache_stats()
        stats['memory_cache_raw'] = len(self._raw_embeddings_cache)
        stats['memory_cache_pca'] = len(self._embedding_cache)
        return stats

    def clear_cache(self, model_name: str = None) -> int:
        """Clear cache entries including PCA models."""
        # Clear memory caches
        if model_name is None or model_name == f"{self.model_name}_raw":
            self._raw_embeddings_cache.clear()
        if model_name is None or model_name == f"{self.model_name}_pca":
            self._embedding_cache.clear()
        
        # Clear PCA cache if clearing all or this model
        if model_name is None or model_name == self.model_name:
            self.db_cache.clear_pca_cache(self.model_name)
            # Reset PCA state
            self.pca = None
            self.pca_fitted = False
        
        # Clear database cache
        return self.db_cache.clear_cache(model_name)

    def _get_image_hash(self, image_path: str) -> str:
        """Generate a hash for caching based on file path and modification time."""
        import os
        try:
            stat = os.stat(image_path)
            hash_string = f"{image_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(hash_string.encode()).hexdigest()
        except Exception:
            return hashlib.md5(image_path.encode()).hexdigest()