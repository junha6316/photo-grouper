import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import hashlib
from PIL import Image

class ImageDeduplicator:
    """
    Identifies and manages duplicate/near-duplicate images based on embeddings and metadata.
    """
    
    def __init__(self, similarity_threshold: float = 0.98):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Minimum similarity score to consider images as duplicates (0.98 = 98% similar)
        """
        self.similarity_threshold = similarity_threshold
        
    def find_duplicates(self, embeddings: Dict[str, np.ndarray], 
                       groups: List[List[str]] = None) -> List[Dict]:
        """
        Find duplicate images within groups or across all images.
        
        Args:
            embeddings: Dictionary mapping image paths to embedding vectors
            groups: Optional list of image groups to search within
            
        Returns:
            List of duplicate sets, each containing:
            {
                'images': [list of duplicate image paths],
                'similarities': similarity matrix for the duplicate set,
                'recommended_keeper': path of recommended image to keep,
                'reason': string explaining why this image was recommended
            }
        """
        if not embeddings:
            return []
        
        duplicate_sets = []
        
        if groups:
            # Find duplicates within each group
            for group in groups:
                if len(group) < 2:
                    continue
                group_duplicates = self._find_duplicates_in_group(group, embeddings)
                duplicate_sets.extend(group_duplicates)
        else:
            # Find duplicates across all images
            all_images = list(embeddings.keys())
            group_duplicates = self._find_duplicates_in_group(all_images, embeddings)
            duplicate_sets.extend(group_duplicates)
        
        return duplicate_sets
    
    def _find_duplicates_in_group(self, image_paths: List[str], 
                                 embeddings: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Find duplicates within a single group of images.
        """
        if len(image_paths) < 2:
            return []
        
        # Filter out images without embeddings
        valid_paths = [path for path in image_paths if path in embeddings]
        if len(valid_paths) < 2:
            return []
        
        # Create embedding matrix
        embedding_matrix = np.array([embeddings[path] for path in valid_paths])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Find connected components of highly similar images
        duplicate_groups = self._find_similarity_clusters(
            valid_paths, similarity_matrix, self.similarity_threshold
        )
        
        # Process each duplicate group
        duplicate_sets = []
        for dup_group in duplicate_groups:
            if len(dup_group) < 2:
                continue
                
            # Extract similarity submatrix for this duplicate group
            indices = [valid_paths.index(path) for path in dup_group]
            sub_similarity = similarity_matrix[np.ix_(indices, indices)]
            
            # Find recommended keeper
            keeper, reason = self._recommend_keeper(dup_group)
            
            duplicate_sets.append({
                'images': dup_group,
                'similarities': sub_similarity,
                'recommended_keeper': keeper,
                'reason': reason,
                'avg_similarity': np.mean(sub_similarity[np.triu_indices_from(sub_similarity, k=1)])
            })
        
        return duplicate_sets
    
    def _find_similarity_clusters(self, image_paths: List[str], 
                                 similarity_matrix: np.ndarray, 
                                 threshold: float) -> List[List[str]]:
        """
        Find clusters of images that are above similarity threshold using Union-Find.
        """
        n = len(image_paths)
        
        # Union-Find data structure
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[py] = px
        
        # Connect similar images
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    union(i, j)
        
        # Group by connected components
        clusters = defaultdict(list)
        for i, path in enumerate(image_paths):
            root = find(i)
            clusters[root].append(path)
        
        # Return only clusters with 2+ images
        return [cluster for cluster in clusters.values() if len(cluster) >= 2]
    
    def _recommend_keeper(self, duplicate_paths: List[str]) -> Tuple[str, str]:
        """
        Recommend which image to keep from a group of duplicates.
        
        Priority:
        1. Highest resolution (width * height)
        2. Largest file size
        3. Most recent modification time
        4. Shortest filename (original vs copy indicators)
        5. Alphabetically first filename
        """
        if len(duplicate_paths) == 1:
            return duplicate_paths[0], "Only image in group"
        
        scores = []
        
        for path in duplicate_paths:
            score = 0
            reason_parts = []
            
            try:
                # File stats
                stat = os.stat(path)
                file_size = stat.st_size
                mod_time = stat.st_mtime
                
                # Image dimensions
                with Image.open(path) as img:
                    width, height = img.size
                    resolution = width * height
                
                filename = os.path.basename(path).lower()
                
                # Score based on resolution (higher is better)
                score += resolution / 1000000  # Normalize to millions of pixels
                if resolution > 2000000:  # > 2MP
                    reason_parts.append("high resolution")
                
                # Score based on file size (larger usually means less compression)
                score += file_size / 1000000  # Normalize to MB
                if file_size > 5000000:  # > 5MB
                    reason_parts.append("large file size")
                
                # Score based on modification time (more recent is better)
                score += mod_time / 1000000000  # Normalize timestamp
                
                # Penalty for likely copy indicators in filename
                copy_indicators = ['copy', 'duplicate', 'dup', '_1', '_2', '(1)', '(2)', 'thumb']
                if any(indicator in filename for indicator in copy_indicators):
                    score -= 10
                    reason_parts.append("likely original (no copy indicators)")
                else:
                    reason_parts.append("no copy indicators in filename")
                
                # Penalty for longer filenames (usually copies have longer names)
                score -= len(filename) / 100
                
                scores.append((score, path, reason_parts))
                
            except Exception as e:
                # If we can't analyze the image, give it a low score
                scores.append((-1000, path, [f"analysis failed: {str(e)}"]))
        
        # Sort by score (highest first)
        scores.sort(reverse=True)
        best_path = scores[0][1]
        reason_parts = scores[0][2]
        
        # Create reason string
        if reason_parts:
            reason = f"Best quality: {', '.join(reason_parts[:3])}"  # Limit to top 3 reasons
        else:
            reason = "Highest overall score"
        
        return best_path, reason
    
    def get_deduplication_stats(self, duplicate_sets: List[Dict]) -> Dict:
        """
        Generate statistics about the deduplication results.
        """
        total_duplicates = sum(len(dup_set['images']) for dup_set in duplicate_sets)
        total_sets = len(duplicate_sets)
        total_redundant = total_duplicates - total_sets  # Images that could be removed
        
        # Calculate potential space savings
        potential_savings = 0
        for dup_set in duplicate_sets:
            keeper = dup_set['recommended_keeper']
            for image_path in dup_set['images']:
                if image_path != keeper:
                    try:
                        potential_savings += os.path.getsize(image_path)
                    except OSError:
                        pass
        
        # Calculate average similarity
        avg_similarity = 0
        if duplicate_sets:
            all_similarities = []
            for dup_set in duplicate_sets:
                all_similarities.append(dup_set.get('avg_similarity', 0))
            avg_similarity = np.mean(all_similarities)
        
        return {
            'total_duplicate_sets': total_sets,
            'total_duplicate_images': total_duplicates,
            'total_redundant_images': total_redundant,
            'potential_space_savings_mb': potential_savings / (1024 * 1024),
            'average_similarity': avg_similarity
        }
    
    def apply_deduplication(self, duplicate_sets: List[Dict], 
                           user_choices: Dict[str, str] = None) -> Dict:
        """
        Apply deduplication by moving/deleting chosen images.
        
        Args:
            duplicate_sets: List of duplicate sets from find_duplicates()
            user_choices: Dict mapping duplicate set index to chosen keeper path
                         If None, uses recommended keepers
        
        Returns:
            Dict with results of the deduplication process
        """
        results = {
            'kept_images': [],
            'removed_images': [],
            'errors': [],
            'space_saved_mb': 0
        }
        
        for i, dup_set in enumerate(duplicate_sets):
            # Determine which image to keep
            if user_choices and str(i) in user_choices:
                keeper = user_choices[str(i)]
            else:
                keeper = dup_set['recommended_keeper']
            
            if keeper not in dup_set['images']:
                results['errors'].append(f"Invalid keeper choice for set {i}: {keeper}")
                continue
            
            results['kept_images'].append(keeper)
            
            # Process images to remove
            for image_path in dup_set['images']:
                if image_path != keeper:
                    try:
                        file_size = os.path.getsize(image_path)
                        results['removed_images'].append(image_path)
                        results['space_saved_mb'] += file_size / (1024 * 1024)
                    except OSError as e:
                        results['errors'].append(f"Error processing {image_path}: {str(e)}")
        
        return results
    
    def create_deduplication_plan(self, duplicate_sets: List[Dict]) -> str:
        """
        Create a human-readable deduplication plan.
        """
        if not duplicate_sets:
            return "No duplicates found."
        
        lines = []
        lines.append(f"Found {len(duplicate_sets)} duplicate sets:")
        lines.append("")
        
        for i, dup_set in enumerate(duplicate_sets, 1):
            lines.append(f"Set {i}: {len(dup_set['images'])} similar images")
            lines.append(f"  Average similarity: {dup_set.get('avg_similarity', 0):.1%}")
            lines.append(f"  Recommended keeper: {os.path.basename(dup_set['recommended_keeper'])}")
            lines.append(f"  Reason: {dup_set['reason']}")
            
            lines.append("  Images:")
            for image_path in dup_set['images']:
                marker = "🏆 KEEP" if image_path == dup_set['recommended_keeper'] else "❌ REMOVE"
                lines.append(f"    {marker} {os.path.basename(image_path)}")
            lines.append("")
        
        stats = self.get_deduplication_stats(duplicate_sets)
        lines.append(f"Summary:")
        lines.append(f"  • {stats['total_redundant_images']} images can be removed")
        lines.append(f"  • Potential space savings: {stats['potential_space_savings_mb']:.1f} MB")
        lines.append(f"  • Average similarity: {stats['average_similarity']:.1%}")
        
        return "\n".join(lines)