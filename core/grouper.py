import random
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from collections import defaultdict
import networkx as nx

# Try to import FAISS for faster similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
    print("FAISS available for accelerated similarity search")
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, using scikit-learn for similarity search")

class PhotoGrouper:
    def __init__(self, tile_size: int = 1000, use_faiss: bool = True):
        """
        Initialize photo grouper.
        
        Args:
            tile_size: Size of tiles for memory-efficient similarity computation
            use_faiss: Whether to use FAISS for acceleration when available
        """
        self.tile_size = tile_size
        self.use_faiss = use_faiss and FAISS_AVAILABLE
    
    def group_by_threshold(self, embeddings: Dict[str, np.ndarray], 
                          threshold: float = 0.85, min_group_size: int = 1) -> Tuple[List[List[str]], List[float]]:
        """
        Group images based on cosine similarity threshold using direct comparison.
        
        Args:
            embeddings: Dictionary mapping image paths to embedding vectors
            threshold: Similarity threshold for grouping (0.0 to 1.0)
            min_group_size: Minimum number of images in a group to include
            
        Returns:
            Tuple of (groups, similarities) where:
            - groups: List of groups, where each group is a list of image paths
            - similarities: List of average similarity scores for each group
        """
        if not embeddings:
            return [], []
        
        image_paths = list(embeddings.keys())
        n = len(image_paths)
        
        if n <= 1:
            return [image_paths], [1.0]  # Single image has perfect similarity with itself
        
        # Use direct similarity algorithm for better results
        return self._group_by_direct_similarity(embeddings, threshold, min_group_size)
    
    def _group_by_direct_similarity(self, embeddings: Dict[str, np.ndarray], 
                                   threshold: float, min_group_size: int) -> Tuple[List[List[str]], List[float]]:
        """
        NetworkX-based grouping algorithm using connected components.
        Creates a graph where edges connect images above similarity threshold,
        then finds connected components as groups.
        """
        # Sort image paths by filename for consistent, predictable results
        image_paths = sorted(embeddings.keys())
        n = len(image_paths)
        
        print(f"Grouping {n} images using NetworkX connected components")
        
        # Create embedding matrix
        embedding_matrix = np.array([embeddings[path] for path in image_paths])
        
        # Compute full similarity matrix
        sim_matrix = cosine_similarity(embedding_matrix)
        print(f"Similarity matrix shape: {sim_matrix.shape}")
        
        # Create NetworkX graph and add all nodes first
        G = nx.Graph()
        G.add_nodes_from(range(n))  # Add all nodes to ensure isolated nodes are included
        
        # Add edges for similar images
        for i in range(n):
            for j in range(i+1, n):
                if sim_matrix[i, j] >= threshold:
                    G.add_edge(i, j)
        
        # Find connected components (includes isolated nodes as single-node components)
        groups = list(nx.connected_components(G))
        
        # Convert sets to lists for consistency
        groups = [list(group) for group in groups]

        # Convert indices to image paths and separate singles from multi-image groups
        multi_groups = []
        multi_similarities = []
        single_images = []
        
        for group_indices in groups:
            if len(group_indices) >= min_group_size:
                group_paths = [image_paths[idx] for idx in group_indices]
                if len(group_paths) == 1:
                    # Collect single images separately
                    single_images.extend(group_paths)
                else:
                    # Calculate average similarity for this group
                    if len(group_indices) > 1:
                        # Get pairwise similarities within the group
                        group_sims = []
                        for i, idx1 in enumerate(group_indices):
                            for idx2 in group_indices[i+1:]:
                                group_sims.append(sim_matrix[idx1, idx2])
                        avg_similarity = np.mean(group_sims) if group_sims else threshold
                    else:
                        avg_similarity = 1.0  # Single image has perfect similarity
                    
                    # Multi-image groups go directly to results
                    multi_groups.append(group_paths)
                    multi_similarities.append(avg_similarity)
        
        # If we have single images, group them together into one "Singles" category
        result_groups = multi_groups
        result_similarities = multi_similarities
        if single_images:
            print(f"Grouping {len(single_images)} singleton images into one 'Singles' category")
            result_groups.append(single_images)
            # Singles don't have meaningful similarity
            result_similarities.append(0.0)
        
        return result_groups, result_similarities
    
    def sort_clusters_by_similarity(self, groups: List[List[str]], 
                                   embeddings: Dict[str, np.ndarray], 
                                   similarities: List[float] = None,
                                   cluster_threshold: float = 0.8) -> Tuple[List[List[str]], List[float]]:
        """
        Sort clusters based on main_image similarity to group related clusters together.
        Uses NetworkX connected components approach for better cluster ordering.
        
        Args:
            groups: List of image groups (clusters)
            embeddings: Dictionary mapping image paths to embeddings
            similarities: List of average similarity scores for each group
            cluster_threshold: Threshold for connecting clusters (default 0.8)
            
        Returns:
            Tuple of (sorted_groups, sorted_similarities) sorted by inter-cluster similarity
        """
        if len(groups) <= 1:
            if similarities:
                return groups, similarities
            else:
                return groups, [1.0] * len(groups)
        
        # Select main_image (representative image) for each cluster
        cluster_main_images = []
        cluster_embeddings = []
        
        for group in groups:
            main_image_path, main_embedding = self._select_main_image(group, embeddings)
            cluster_main_images.append(main_image_path)
            cluster_embeddings.append(main_embedding)
        
        # Compute similarity matrix between main_image embeddings
        main_embedding_matrix = np.array(cluster_embeddings)
        cluster_similarity = cosine_similarity(main_embedding_matrix)
        # save cluster_similarity to file
        # np.save("cluster_similarity.npy", cluster_similarity)
        
        # Apply NetworkX connected components approach for cluster ordering
        G = nx.Graph()
        # Add all nodes first to ensure isolated nodes are included
        G.add_nodes_from(range(cluster_similarity.shape[0]))
        
        for i in range(cluster_similarity.shape[0]):
            for j in range(i+1, cluster_similarity.shape[0]):
                if cluster_similarity[i, j] >= cluster_threshold:
                    G.add_edge(i, j)
        
        # Get connected components and sort them (includes isolated nodes)
        components = list(nx.connected_components(G))
        components = [list(comp) for comp in components]
        
        # Sort components by size (largest first) and flatten
        components.sort(key=len, reverse=True)
        sorted_indices = []
        for component in components:
            # Within each component, sort clusters by size (largest groups first)
            component_with_sizes = [(idx, len(groups[idx])) for idx in component]
            component_with_sizes.sort(key=lambda x: x[1], reverse=True)
            sorted_indices.extend([idx for idx, _ in component_with_sizes])
        
        # Reorder groups and similarities based on sorted indices
        sorted_groups = [groups[i] for i in sorted_indices]
        sorted_similarities = [similarities[i] for i in sorted_indices] if similarities else [1.0] * len(sorted_groups)
        
        # Find and move singles group to the front
        # The singles group is typically the largest group containing many unrelated images
        singles_group_idx = None
        max_size = 0
        
        for i, group in enumerate(sorted_groups):
            # Heuristic: singles group is usually the largest and has low internal similarity
            if len(group) > max_size and len(group) > 10:  # At least 10 images to be considered singles
                # Check if this is likely a singles group by sampling similarity
                sample_size = max(10, len(group))
                sample_paths = random.sample(group, sample_size)
                sample_embeddings = [embeddings[path] for path in sample_paths if path in embeddings]
                
                if len(sample_embeddings) >= 2:
                    sample_matrix = np.array(sample_embeddings)
                    sample_similarity = cosine_similarity(sample_matrix)
                    avg_similarity = np.mean(sample_similarity[np.triu_indices_from(sample_similarity, k=1)])
                    
                    # If average similarity is low, it's likely the singles group
                    if avg_similarity < 0.5:  # Low similarity threshold
                        max_size = len(group)
                        singles_group_idx = i
        print(f"DEBUG: singles_group_idx: {singles_group_idx}")
        # Move singles group to front if found
        if singles_group_idx is not None and singles_group_idx > 0:
            
            singles_group = sorted_groups.pop(singles_group_idx)
            singles_similarity = sorted_similarities.pop(singles_group_idx)
            sorted_groups = [singles_group] + sorted_groups
            sorted_similarities = [singles_similarity] + sorted_similarities
            print(f"Moved singles group ({len(singles_group)} images) to front")
        
        # Debug output to show main images selected
        print("Cluster main images selected:")
        for i, group in enumerate(sorted_groups):
            main_image_path = self._select_main_image(group, embeddings)[0]
            print(f"  Cluster {i+1}: {os.path.basename(main_image_path)} ({len(group)} images)")
        
        return sorted_groups, sorted_similarities
    
    def _select_main_image(self, group: List[str], embeddings: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray]:
        """
        Select the most representative main_image from a cluster.
        
        Strategy: Choose the image that has the highest average similarity to all other images in the cluster.
        This gives us the most "central" image that best represents the cluster.
        
        Args:
            group: List of image paths in the cluster
            embeddings: Dictionary mapping image paths to embeddings
            
        Returns:
            Tuple of (main_image_path, main_image_embedding)
        """
        if len(group) == 1:
            # Single image cluster - it's the main image by default
            path = group[0]
            return path, embeddings[path]
        
        # Get embeddings for all images in the group
        group_embeddings = []
        valid_paths = []
        
        for path in group:
            if path in embeddings:
                group_embeddings.append(embeddings[path])
                valid_paths.append(path)
        
        if not group_embeddings:
            # Fallback: return first image if no embeddings found
            return group[0], np.zeros(len(next(iter(embeddings.values()))))
        
        if len(group_embeddings) == 1:
            # Only one valid embedding
            return valid_paths[0], group_embeddings[0]
        
        # Calculate similarity matrix for the group
        embedding_matrix = np.array(group_embeddings)
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Find the image with highest average similarity to all others
        # (excluding self-similarity which is always 1.0)
        avg_similarities = []
        for i in range(len(group_embeddings)):
            # Calculate average similarity excluding self (diagonal = 1.0)
            similarities = similarity_matrix[i]
            avg_sim = (np.sum(similarities) - 1.0) / (len(similarities) - 1)
            avg_similarities.append(avg_sim)
        
        # Select image with highest average similarity as main_image
        main_idx = np.argmax(avg_similarities)
        main_path = valid_paths[main_idx]
        main_embedding = group_embeddings[main_idx]
    
        
        return main_path, main_embedding