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

class UnionFind:
    """Union-Find data structure for grouping connected components."""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

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
                          threshold: float = 0.85, min_group_size: int = 1) -> List[List[str]]:
        """
        Group images based on cosine similarity threshold using direct comparison.
        
        Args:
            embeddings: Dictionary mapping image paths to embedding vectors
            threshold: Similarity threshold for grouping (0.0 to 1.0)
            min_group_size: Minimum number of images in a group to include
            
        Returns:
            List of groups, where each group is a list of image paths
        """
        if not embeddings:
            return []
        
        image_paths = list(embeddings.keys())
        n = len(image_paths)
        
        if n <= 1:
            return [image_paths]
        
        # Use direct similarity algorithm for better results
        return self._group_by_direct_similarity(embeddings, threshold, min_group_size)
    
    def _group_by_direct_similarity(self, embeddings: Dict[str, np.ndarray], 
                                   threshold: float, min_group_size: int) -> List[List[str]]:
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
        single_images = []
        
        for group_indices in groups:
            if len(group_indices) >= min_group_size:
                group_paths = [image_paths[idx] for idx in group_indices]
                if len(group_paths) == 1:
                    # Collect single images separately
                    single_images.extend(group_paths)
                else:
                    # Multi-image groups go directly to results
                    multi_groups.append(group_paths)
        
        # If we have single images, group them together into one "Singles" category
        result_groups = multi_groups
        if single_images:
            print(f"Grouping {len(single_images)} singleton images into one 'Singles' category")
            result_groups.append(single_images)
        
        return result_groups
    
    def sort_clusters_by_similarity(self, groups: List[List[str]], 
                                   embeddings: Dict[str, np.ndarray], 
                                   cluster_threshold: float = 0.8) -> List[List[str]]:
        """
        Sort clusters based on main_image similarity to group related clusters together.
        Uses NetworkX connected components approach for better cluster ordering.
        
        Args:
            groups: List of image groups (clusters)
            embeddings: Dictionary mapping image paths to embeddings
            cluster_threshold: Threshold for connecting clusters (default 0.8)
            
        Returns:
            Reordered list of groups sorted by inter-cluster similarity
        """
        if len(groups) <= 1:
            return groups
        
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
        
        # Reorder groups based on sorted indices
        sorted_groups = [groups[i] for i in sorted_indices]
        
        # Find and move singles group to the front
        # The singles group is typically the largest group containing many unrelated images
        singles_group_idx = None
        max_size = 0
        
        for i, group in enumerate(sorted_groups):
            # Heuristic: singles group is usually the largest and has low internal similarity
            if len(group) > max_size and len(group) > 10:  # At least 10 images to be considered singles
                # Check if this is likely a singles group by sampling similarity
                sample_size = min(5, len(group))
                sample_paths = group[:sample_size]
                sample_embeddings = [embeddings[path] for path in sample_paths if path in embeddings]
                
                if len(sample_embeddings) >= 2:
                    sample_matrix = np.array(sample_embeddings)
                    sample_similarity = cosine_similarity(sample_matrix)
                    avg_similarity = np.mean(sample_similarity[np.triu_indices_from(sample_similarity, k=1)])
                    
                    # If average similarity is low, it's likely the singles group
                    if avg_similarity < 0.6:  # Low similarity threshold
                        max_size = len(group)
                        singles_group_idx = i
        
        # Move singles group to front if found
        if singles_group_idx is not None and singles_group_idx > 0:
            singles_group = sorted_groups.pop(singles_group_idx)
            sorted_groups.insert(0, singles_group)
            print(f"Moved singles group ({len(singles_group)} images) to front")
        
        # Debug output to show main images selected
        print("Cluster main images selected:")
        for i, group in enumerate(sorted_groups):
            main_image_path = self._select_main_image(group, embeddings)[0]
            print(f"  Cluster {i+1}: {os.path.basename(main_image_path)} ({len(group)} images)")
        
        return sorted_groups
    
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
        
        print(f"  Main image for cluster: {os.path.basename(main_path)} (avg_sim: {avg_similarities[main_idx]:.3f})")
        
        return main_path, main_embedding
    
    def _sort_by_similarity_greedy(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Greedy algorithm to sort clusters by similarity.
        Starts with the first cluster and picks the most similar unvisited cluster next.
        If max similarity drops below 0.5, switches to a new reference cluster.
        """
        n = len(similarity_matrix)
        if n <= 1:
            return list(range(n))
        
        visited = set()
        sorted_indices = []
        
        # Start with the first cluster
        current = 0
        sorted_indices.append(current)
        visited.add(current)
        
        # Greedily select the most similar unvisited cluster
        while len(sorted_indices) < n:
            best_similarity = -1
            best_idx = None
            
            # Find the most similar unvisited cluster to the current one
            for i in range(n):
                if i not in visited:
                    sim = similarity_matrix[current, i]
                    if sim > best_similarity:
                        best_similarity = sim
                        best_idx = i
            
            # If max similarity is too low (≤ 0.5), switch to a new reference cluster
            if best_similarity <= 0.5:
                print(f"Low similarity ({best_similarity:.3f}) detected. Switching reference cluster.")
                new_reference = self._find_best_reference_cluster(similarity_matrix, visited)
                if new_reference is not None:
                    current = new_reference
                    sorted_indices.append(current)
                    visited.add(current)
                    print(f"Switched to new reference cluster {current}")
                    continue
            
            if best_idx is not None:
                sorted_indices.append(best_idx)
                visited.add(best_idx)
                current = best_idx
            else:
                # Fallback: add any remaining unvisited clusters
                for i in range(n):
                    if i not in visited:
                        sorted_indices.append(i)
                        visited.add(i)
                break
        
        return sorted_indices
    
    def _find_best_reference_cluster(self, similarity_matrix: np.ndarray, visited: set) -> int:
        """
        Find the best new reference cluster when current reference has low similarity.
        
        Strategy: Choose the unvisited cluster that has the highest average similarity 
        to all other unvisited clusters.
        
        Args:
            similarity_matrix: Cluster similarity matrix
            visited: Set of already visited cluster indices
            
        Returns:
            Index of the best new reference cluster, or None if no good candidate
        """
        unvisited = [i for i in range(len(similarity_matrix)) if i not in visited]
        
        if not unvisited:
            return None
        
        if len(unvisited) == 1:
            return unvisited[0]
        
        best_cluster = None
        best_avg_similarity = -1
        
        # For each unvisited cluster, calculate its average similarity to other unvisited clusters
        for candidate in unvisited:
            similarities = []
            for other in unvisited:
                if other != candidate:
                    similarities.append(similarity_matrix[candidate, other])
            
            if similarities:  # Avoid division by zero
                avg_similarity = np.mean(similarities)
                if avg_similarity > best_avg_similarity:
                    best_avg_similarity = avg_similarity
                    best_cluster = candidate
        
        print(f"Best new reference cluster {best_cluster} with avg similarity {best_avg_similarity:.3f}")
        return best_cluster
    
    def _sort_by_hybrid_approach(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Hybrid approach combining connectivity analysis with adaptive sorting.
        
        1. Find connected components based on similarity threshold
        2. Sort components by size (largest first)  
        3. Within each component: use greedy if small, spectral if large
        4. Between components: arrange by average inter-component similarity
        """
        n = len(similarity_matrix)
        if n <= 1:
            return list(range(n))
        
        print("Using hybrid clustering sort approach")
        
        # Step 1: Find connected components
        connected_components = self._find_connected_components(similarity_matrix, threshold=0.3)
        print(f"Found {len(connected_components)} connected components")
        
        if len(connected_components) == 1:
            # All clusters are connected: use spectral ordering for smooth transitions
            print("Single connected component detected, using spectral ordering")
            return self._sort_by_spectral_ordering(similarity_matrix)
        else:
            # Multiple components: sort each separately, then arrange by inter-component similarity
            print("Multiple components detected, using component-wise sorting")
            return self._sort_multiple_components(similarity_matrix, connected_components)
    
    def _find_connected_components(self, similarity_matrix: np.ndarray, 
                                   threshold: float = 0.3) -> List[List[int]]:
        """Find connected components based on similarity threshold."""
        n = len(similarity_matrix)
        adjacency = similarity_matrix >= threshold
        np.fill_diagonal(adjacency, False)  # Remove self-connections
        
        visited = set()
        components = []
        
        for i in range(n):
            if i not in visited:
                component = []
                self._dfs_component(adjacency, i, visited, component)
                components.append(component)
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {len(comp)} clusters")
        
        return components
    
    def _dfs_component(self, adjacency: np.ndarray, node: int, 
                       visited: set, component: list):
        """DFS to find connected component."""
        visited.add(node)
        component.append(node)
        
        for neighbor in range(len(adjacency)):
            if adjacency[node, neighbor] and neighbor not in visited:
                self._dfs_component(adjacency, neighbor, visited, component)
    
    def _sort_multiple_components(self, similarity_matrix: np.ndarray, 
                                  components: List[List[int]]) -> List[int]:
        """Sort multiple components and arrange them optimally."""
        if len(components) == 1:
            return self._sort_single_component(similarity_matrix, components[0])
        
        # Sort each component internally
        sorted_components = []
        for i, component in enumerate(components):
            sorted_comp = self._sort_single_component(similarity_matrix, component)
            sorted_components.append(sorted_comp)
            print(f"  Sorted component {i+1} internally")
        
        # Arrange components by inter-component similarity
        component_order = self._arrange_components_by_similarity(
            similarity_matrix, sorted_components
        )
        
        # Flatten to final order
        final_order = []
        for comp_idx in component_order:
            final_order.extend(sorted_components[comp_idx])
        
        return final_order
    
    def _sort_single_component(self, similarity_matrix: np.ndarray, 
                               component: List[int]) -> List[int]:
        """Sort a single component using the best method for its size."""
        if len(component) <= 1:
            return component
        
        if len(component) <= 5:
            # Small component: use enhanced greedy
            sub_matrix = similarity_matrix[np.ix_(component, component)]
            sub_order = self._sort_by_similarity_greedy(sub_matrix)
            return [component[i] for i in sub_order]
        else:
            # Large component: use spectral ordering for smoother transitions
            sub_matrix = similarity_matrix[np.ix_(component, component)]
            sub_order = self._sort_by_spectral_ordering(sub_matrix)
            return [component[i] for i in sub_order]
    
    def _arrange_components_by_similarity(self, similarity_matrix: np.ndarray,
                                          sorted_components: List[List[int]]) -> List[int]:
        """Arrange components to minimize dissimilarity between adjacent components."""
        n_components = len(sorted_components)
        if n_components <= 1:
            return list(range(n_components))
        
        # Calculate inter-component similarities
        inter_comp_sim = np.zeros((n_components, n_components))
        
        for i in range(n_components):
            for j in range(i + 1, n_components):
                # Average similarity between all pairs of clusters in different components
                comp_i = sorted_components[i]
                comp_j = sorted_components[j]
                
                similarities = []
                for ci in comp_i:
                    for cj in comp_j:
                        similarities.append(similarity_matrix[ci, cj])
                
                avg_sim = np.mean(similarities) if similarities else 0.0
                inter_comp_sim[i, j] = avg_sim
                inter_comp_sim[j, i] = avg_sim
        
        # Use greedy approach to order components
        visited = set()
        component_order = []
        
        # Start with the component that has highest average similarity to others
        avg_similarities = np.mean(inter_comp_sim, axis=1)
        current = np.argmax(avg_similarities)
        component_order.append(current)
        visited.add(current)
        
        print(f"Starting component arrangement with component {current} (avg_sim: {avg_similarities[current]:.3f})")
        
        # Greedily add most similar unvisited component
        while len(component_order) < n_components:
            best_sim = -1
            best_comp = None
            
            for i in range(n_components):
                if i not in visited:
                    sim = inter_comp_sim[current, i]
                    if sim > best_sim:
                        best_sim = sim
                        best_comp = i
            
            if best_comp is not None:
                component_order.append(best_comp)
                visited.add(best_comp)
                current = best_comp
                print(f"  Added component {best_comp} (similarity: {best_sim:.3f})")
            else:
                # Add remaining components
                for i in range(n_components):
                    if i not in visited:
                        component_order.append(i)
                        visited.add(i)
                break
        
        return component_order
    
    def _sort_by_spectral_ordering(self, similarity_matrix: np.ndarray) -> List[int]:
        """
        Use spectral ordering for smooth 1D arrangement of clusters.
        Based on the Fiedler vector of the graph Laplacian.
        """
        n = len(similarity_matrix)
        if n <= 1:
            return list(range(n))
        
        try:
            # Ensure matrix is symmetric and non-negative
            sim_matrix = np.maximum(similarity_matrix, similarity_matrix.T)
            sim_matrix = np.maximum(sim_matrix, 0)
            
            # Compute degree matrix and Laplacian
            degrees = np.sum(sim_matrix, axis=1)
            # Add small epsilon to avoid division by zero
            degrees = np.maximum(degrees, 1e-10)
            
            degree_matrix = np.diag(degrees)
            laplacian = degree_matrix - sim_matrix
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # Use Fiedler vector (eigenvector of second smallest eigenvalue)
            fiedler_vector = eigenvectors[:, 1]
            
            # Sort by Fiedler vector values
            sorted_indices = np.argsort(fiedler_vector)
            
            print(f"Spectral ordering completed (Fiedler vector range: {fiedler_vector.min():.3f} to {fiedler_vector.max():.3f})")
            return sorted_indices.tolist()
            
        except Exception as e:
            print(f"Spectral ordering failed ({e}), falling back to greedy")
            return self._sort_by_similarity_greedy(similarity_matrix)
    
    def _group_by_threshold_faiss(self, embeddings: Dict[str, np.ndarray], 
                                 threshold: float) -> List[List[str]]:
        """FAISS-accelerated grouping for large datasets."""
        image_paths = list(embeddings.keys())
        n = len(image_paths)
        
        # Create embedding matrix
        embedding_matrix = np.array([embeddings[path] for path in image_paths]).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Build FAISS index
        dimension = embedding_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        index.add(embedding_matrix)
        
        # Initialize Union-Find
        uf = UnionFind(n)
        
        # Search for similar embeddings for each image
        batch_size = min(1000, n)
        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            query_embeddings = embedding_matrix[i:end_i]
            
            # Search for all similar embeddings above threshold
            similarities, indices = index.search(query_embeddings, min(n, 1000))
            
            for local_idx, (sims, ids) in enumerate(zip(similarities, indices)):
                global_idx = i + local_idx
                for sim, idx in zip(sims, ids):
                    if sim >= threshold and idx != global_idx and idx < n:
                        uf.union(global_idx, int(idx))
        
        # Build groups from Union-Find components
        groups = defaultdict(list)
        for i, path in enumerate(image_paths):
            root = uf.find(i)
            groups[root].append(path)
        
        # Convert to list and filter out single-image groups for cleaner output
        result_groups = [group for group in groups.values() if len(group) > 1]
        
        # Add singleton groups (images with no similar matches)
        singleton_groups = [group for group in groups.values() if len(group) == 1]
        result_groups.extend(singleton_groups)
        
        return result_groups
    
    def _group_by_threshold_sklearn(self, embeddings: Dict[str, np.ndarray], 
                                   threshold: float) -> List[List[str]]:
        """Original scikit-learn based grouping with tiling."""
        image_paths = list(embeddings.keys())
        n = len(image_paths)
        
        # Create embedding matrix
        embedding_matrix = np.array([embeddings[path] for path in image_paths])
        
        # Initialize Union-Find
        uf = UnionFind(n)
        
        # Compute similarities in tiles to save memory
        for i in range(0, n, self.tile_size):
            for j in range(i, n, self.tile_size):
                # Get tile boundaries
                i_end = min(i + self.tile_size, n)
                j_end = min(j + self.tile_size, n)
                
                # Compute similarity for this tile
                tile_i = embedding_matrix[i:i_end]
                tile_j = embedding_matrix[j:j_end]
                similarities = cosine_similarity(tile_i, tile_j)
                
                # Find pairs above threshold
                for local_i, global_i in enumerate(range(i, i_end)):
                    j_start = max(0, global_i - j) if j <= global_i else 0
                    for local_j, global_j in enumerate(range(j + j_start, j_end)):
                        if global_i != global_j and similarities[local_i, local_j] >= threshold:
                            uf.union(global_i, global_j)
        
        # Build groups from Union-Find components
        groups = defaultdict(list)
        for i, path in enumerate(image_paths):
            root = uf.find(i)
            groups[root].append(path)
        
        # Convert to list and filter out single-image groups for cleaner output
        result_groups = [group for group in groups.values() if len(group) > 1]
        
        # Add singleton groups (images with no similar matches)
        singleton_groups = [group for group in groups.values() if len(group) == 1]
        result_groups.extend(singleton_groups)
        
        return result_groups
    
    def get_similarity_matrix(self, embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute full similarity matrix for debugging/visualization.
        
        Args:
            embeddings: Dictionary mapping image paths to embeddings
            
        Returns:
            Tuple of (similarity_matrix, image_paths)
        """
        if not embeddings:
            return np.array([]), []
        
        image_paths = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[path] for path in image_paths])
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        return similarity_matrix, image_paths