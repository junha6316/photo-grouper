import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from infra.cache_db import EmbeddingCache

# Initialize HEIF support if available
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


DEFAULT_FOCUS_THRESHOLD = 100.0
DEFAULT_MAX_SIZE = 512
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 4))


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
    )
    return float(np.var(laplacian))


def compute_focus_score(image_path: str, max_size: int = DEFAULT_MAX_SIZE) -> float:
    try:
        with Image.open(image_path) as image:
            image = image.convert("L")
            if max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            gray = np.asarray(image, dtype=np.float32)
        return _laplacian_variance(gray)
    except Exception as exc:
        print(f"Focus score error for {image_path}: {exc}")
        return 0.0


def compute_focus_scores(
    image_paths: List[str],
    max_size: int = DEFAULT_MAX_SIZE,
    max_workers: Optional[int] = None,
    progress_callback=None,
    use_cache: bool = True,
) -> Dict[str, float]:
    if not image_paths:
        return {}

    scores: Dict[str, float] = {}
    new_scores: Dict[str, float] = {}
    worker_count = max_workers or DEFAULT_WORKERS
    total = len(image_paths)
    completed = 0

    cache = EmbeddingCache() if use_cache else None
    if cache:
        cached_scores = cache.get_focus_scores(image_paths)
        for path, score in cached_scores.items():
            scores[path] = score
            completed += 1
            if progress_callback:
                progress_callback(completed, total, path)

    remaining_paths = [path for path in image_paths if path not in scores]
    if not remaining_paths:
        return scores

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(compute_focus_score, path, max_size): path
            for path in remaining_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                score = future.result()
            except Exception as exc:
                print(f"Focus score error for {path}: {exc}")
                score = 0.0
            scores[path] = score
            new_scores[path] = score
            completed += 1
            if progress_callback:
                progress_callback(completed, total, path)

    if cache and new_scores:
        cache.save_focus_scores(new_scores)

    return scores
