import os
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from utils.math_utils import pairwise_cosine_similarity


class PhotoGrouper:
    def __init__(self, tile_size: int = 1000, use_faiss: bool = False, verbose: bool = True):
        """
        Initialize photo grouper.
        
        Args:
            tile_size: Size of tiles for memory-efficient similarity computation
            use_faiss: Whether to use FAISS for acceleration when available
            verbose: Whether to print debug information during grouping
        """
        self.tile_size = tile_size
        self.use_faiss = use_faiss
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
    
    def group_by_threshold(self, embeddings: Dict[str, np.ndarray],
                          threshold: float = 0.9, min_group_size: int = 1,
                          coarse_threshold: float = 0.70,
                          strict_mode: str = "min") -> Tuple[List[List[str]], List[float]]:
        """
        Group images using a coarse-to-fine strategy.
        First, build coarse components using `coarse_threshold`, then refine
        each component with a stricter threshold using `strict_mode`.
        
        Args:
            embeddings: Dictionary mapping image paths to embedding vectors
            threshold: Similarity threshold for grouping (0.0 to 1.0)
            min_group_size: Minimum number of images in a group to include
            coarse_threshold: Coarse similarity threshold used for initial components
            strict_mode: "avg" (default) or "min" to control refinement strictness
            
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
        
        # Use coarse-to-fine similarity algorithm for better results
        return self._group_by_direct_similarity(
            embeddings,
            threshold,
            min_group_size,
            coarse_threshold,
            strict_mode,
        )
    
    def _group_by_direct_similarity(self, embeddings: Dict[str, np.ndarray],
                                   threshold: float, min_group_size: int,
                                   coarse_threshold: float,
                                   strict_mode: str) -> Tuple[List[List[str]], List[float]]:
        """
        Coarse-to-fine grouping using connected components for the coarse step,
        then medoid-based refinement within each component.
        """
        # Sort image paths by filename for consistent, predictable results
        image_paths = sorted(embeddings.keys())
        n = len(image_paths)
        
        self._log(f"Grouping {n} images using coarse-to-fine components")
        
        # Create embedding matrix
        embedding_matrix = np.array([embeddings[path] for path in image_paths])
        
        # Compute full similarity matrix
        sim_matrix = pairwise_cosine_similarity(embedding_matrix)
        self._log(f"Similarity matrix shape: {sim_matrix.shape}")

        # If strict threshold is already loose, fall back to direct components
        effective_coarse = min(coarse_threshold, threshold)
        if threshold <= effective_coarse:
            return self._group_by_components(
                sim_matrix,
                image_paths,
                threshold,
                min_group_size,
            )
        
        # Coarse components at 0.70 (or lower if strict is lower)
        coarse_components = self._components_from_similarity(sim_matrix, effective_coarse)

        # Refine each coarse component using the strict threshold
        refined_indices = []
        for component in coarse_components:
            if len(component) <= 1:
                refined_indices.append(component)
                continue
            refined_indices.extend(
                self._refine_component_by_medoid(
                    sim_matrix,
                    component,
                    threshold,
                    strict_mode,
                )
            )

        return self._indices_to_groups(
            refined_indices,
            image_paths,
            sim_matrix,
            threshold,
            min_group_size,
        )

    def _components_from_similarity(self, sim_matrix: np.ndarray,
                                    threshold: float) -> List[List[int]]:
        """Build connected components using a similarity threshold."""
        n = sim_matrix.shape[0]
        G = nx.Graph()
        G.add_nodes_from(range(n))  # Include isolated nodes

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    G.add_edge(i, j)

        components = list(nx.connected_components(G))
        return [sorted(list(component)) for component in components]

    def _refine_component_by_medoid(self, sim_matrix: np.ndarray,
                                    component_indices: List[int],
                                    threshold: float,
                                    strict_mode: str) -> List[List[int]]:
        """Split a coarse component using medoid-based refinement."""
        remaining = set(component_indices)
        refined = []

        while remaining:
            remaining_list = sorted(remaining)
            if len(remaining_list) == 1:
                refined.append(remaining_list)
                break

            sub_sim = sim_matrix[np.ix_(remaining_list, remaining_list)]
            avg_sims = (np.sum(sub_sim, axis=1) - 1.0) / (len(remaining_list) - 1)
            medoid_idx = remaining_list[int(np.argmax(avg_sims))]

            cluster = [
                idx for idx in remaining_list
                if sim_matrix[medoid_idx, idx] >= threshold
            ]
            if not cluster:
                cluster = [medoid_idx]

            cluster = self._prune_cluster_by_threshold(
                sim_matrix,
                cluster,
                threshold,
                strict_mode,
                medoid_idx,
            )
            if not cluster:
                cluster = [medoid_idx]

            for idx in cluster:
                remaining.discard(idx)
            refined.append(sorted(cluster))

        return refined

    def _prune_cluster_by_threshold(self, sim_matrix: np.ndarray,
                                    cluster_indices: List[int],
                                    threshold: float,
                                    strict_mode: str,
                                    anchor_idx: int) -> List[int]:
        """Prune cluster members until strict criteria are satisfied."""
        if len(cluster_indices) <= 1:
            return cluster_indices

        cluster = list(dict.fromkeys(cluster_indices))
        if anchor_idx not in cluster:
            cluster.append(anchor_idx)

        while len(cluster) > 1:
            sub_sim = sim_matrix[np.ix_(cluster, cluster)]
            if strict_mode == "min":
                metrics = np.min(sub_sim + np.eye(len(cluster)), axis=1)
            else:
                metrics = (np.sum(sub_sim, axis=1) - 1.0) / (len(cluster) - 1)

            if np.all(metrics >= threshold):
                break

            removal_candidates = [
                (metrics[i], cluster[i])
                for i in range(len(cluster))
                if cluster[i] != anchor_idx
            ]
            if not removal_candidates:
                break

            _, remove_idx = min(removal_candidates, key=lambda x: (x[0], x[1]))
            cluster.remove(remove_idx)

        return cluster

    def _indices_to_groups(self, index_groups: List[List[int]],
                           image_paths: List[str],
                           sim_matrix: np.ndarray,
                           threshold: float,
                           min_group_size: int) -> Tuple[List[List[str]], List[float]]:
        """Convert index groups to paths, compute similarities, and group singles."""
        multi_groups = []
        multi_similarities = []
        single_images = []

        for group_indices in index_groups:
            if len(group_indices) < min_group_size:
                continue

            group_paths = [image_paths[idx] for idx in group_indices]
            if len(group_paths) == 1:
                single_images.extend(group_paths)
                continue

            avg_similarity = self._average_similarity(sim_matrix, group_indices)
            multi_groups.append(group_paths)
            multi_similarities.append(avg_similarity if avg_similarity is not None else threshold)

        result_groups = multi_groups
        result_similarities = multi_similarities
        if single_images:
            self._log(f"Grouping {len(single_images)} singleton images into one 'Singles' category")
            result_groups.append(single_images)
            result_similarities.append(0.0)

        return result_groups, result_similarities

    def _average_similarity(self, sim_matrix: np.ndarray,
                            indices: List[int]) -> float:
        """Compute average pairwise similarity for a group."""
        if len(indices) <= 1:
            return 1.0

        sub_sim = sim_matrix[np.ix_(indices, indices)]
        upper = np.triu_indices_from(sub_sim, k=1)
        if upper[0].size == 0:
            return 1.0

        return float(np.mean(sub_sim[upper]))

    def _group_by_components(self, sim_matrix: np.ndarray,
                             image_paths: List[str],
                             threshold: float,
                             min_group_size: int) -> Tuple[List[List[str]], List[float]]:
        """Fallback: group by connected components at a single threshold."""
        components = self._components_from_similarity(sim_matrix, threshold)
        return self._indices_to_groups(
            components,
            image_paths,
            sim_matrix,
            threshold,
            min_group_size,
        )

    def _cluster_representatives(
        self,
        groups: List[List[str]],
        embeddings: Dict[str, np.ndarray],
    ) -> Tuple[List[str], List[np.ndarray]]:
        cluster_main_images: List[str] = []
        cluster_embeddings: List[np.ndarray] = []
        for group in groups:
            main_image_path, main_embedding = self._select_main_image(group, embeddings)
            cluster_main_images.append(main_image_path)
            cluster_embeddings.append(main_embedding)
        return cluster_main_images, cluster_embeddings

    def _sorted_cluster_indices(
        self,
        cluster_similarity: np.ndarray,
        groups: List[List[str]],
        cluster_threshold: float,
    ) -> List[int]:
        components = self._components_from_similarity(cluster_similarity, cluster_threshold)
        components.sort(key=len, reverse=True)

        sorted_indices: List[int] = []
        for component in components:
            component_with_sizes = [(idx, len(groups[idx])) for idx in component]
            component_with_sizes.sort(key=lambda x: x[1], reverse=True)
            sorted_indices.extend([idx for idx, _ in component_with_sizes])
        return sorted_indices

    def _find_singles_group_index(
        self,
        groups: List[List[str]],
        embeddings: Dict[str, np.ndarray],
    ) -> Optional[int]:
        singles_group_idx = None
        max_size = 0

        for i, group in enumerate(groups):
            if len(group) > max_size and len(group) > 10:
                sample_size = max(10, len(group))
                sample_paths = random.sample(group, sample_size)
                sample_embeddings = [embeddings[path] for path in sample_paths if path in embeddings]

                if len(sample_embeddings) >= 2:
                    sample_matrix = np.array(sample_embeddings)
                    sample_similarity = pairwise_cosine_similarity(sample_matrix)
                    avg_similarity = np.mean(sample_similarity[np.triu_indices_from(sample_similarity, k=1)])

                    if avg_similarity < 0.5:
                        max_size = len(group)
                        singles_group_idx = i

        return singles_group_idx
    
    def sort_clusters_by_similarity(
        self,
        groups: List[List[str]],
        embeddings: Dict[str, np.ndarray],
        similarities: Optional[List[float]] = None,
        cluster_threshold: float = 0.8,
    ) -> Tuple[List[List[str]], List[float]]:
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
            return groups, [1.0] * len(groups)

        cluster_main_images, cluster_embeddings = self._cluster_representatives(groups, embeddings)

        main_embedding_matrix = np.array(cluster_embeddings)
        cluster_similarity = pairwise_cosine_similarity(main_embedding_matrix)
        sorted_indices = self._sorted_cluster_indices(cluster_similarity, groups, cluster_threshold)

        sorted_groups = [groups[i] for i in sorted_indices]
        sorted_similarities = [similarities[i] for i in sorted_indices] if similarities else [1.0] * len(sorted_groups)
        sorted_main_images = [cluster_main_images[i] for i in sorted_indices]

        singles_group_idx = self._find_singles_group_index(sorted_groups, embeddings)
        self._log(f"DEBUG: singles_group_idx: {singles_group_idx}")
        # Move singles group to front if found
        if singles_group_idx is not None and singles_group_idx > 0:
            singles_group = sorted_groups.pop(singles_group_idx)
            singles_similarity = sorted_similarities.pop(singles_group_idx)
            singles_main_image = sorted_main_images.pop(singles_group_idx)
            sorted_groups = [singles_group] + sorted_groups
            sorted_similarities = [singles_similarity] + sorted_similarities
            sorted_main_images = [singles_main_image] + sorted_main_images
            self._log(f"Moved singles group ({len(singles_group)} images) to front")
        
        # Debug output to show main images selected
        self._log("Cluster main images selected:")
        for i, (group, main_image_path) in enumerate(zip(sorted_groups, sorted_main_images)):
            self._log(f"  Cluster {i+1}: {os.path.basename(main_image_path)} ({len(group)} images)")
        
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
        similarity_matrix = pairwise_cosine_similarity(embedding_matrix)
        
        # Find the image with highest average similarity to all others
        # (excluding self-similarity which is always 1.0)
        avg_similarities = (np.sum(similarity_matrix, axis=1) - 1.0) / (len(similarity_matrix) - 1)
        
        # Select image with highest average similarity as main_image
        main_idx = int(np.argmax(avg_similarities))
        main_path = valid_paths[main_idx]
        main_embedding = group_embeddings[main_idx]

        return main_path, main_embedding
