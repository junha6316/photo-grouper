from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional
import threading

import numpy as np
from PIL import Image

DEFAULT_PHASH_DISTANCE = 4
DEFAULT_MAX_GROUP_SIZE = 200

_DCT_CACHE: Dict[int, np.ndarray] = {}
_PHASH_CACHE: Dict[str, Optional[int]] = {}
_CACHE_LOCK = threading.Lock()


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self._rank[px] < self._rank[py]:
            self._parent[px] = py
        elif self._rank[px] > self._rank[py]:
            self._parent[py] = px
        else:
            self._parent[py] = px
            self._rank[px] += 1

    def clusters(self) -> List[List[int]]:
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self._parent)):
            clusters[self.find(idx)].append(idx)

        cluster_entries = []
        for indices in clusters.values():
            indices.sort()
            cluster_entries.append((indices[0], indices))
        cluster_entries.sort(key=lambda item: item[0])
        return [indices for _, indices in cluster_entries]


def _dct_matrix(n: int) -> np.ndarray:
    cached = _DCT_CACHE.get(n)
    if cached is not None:
        return cached

    mat = np.zeros((n, n), dtype=np.float32)
    factor = np.pi / (2 * n)
    for k in range(n):
        scale = np.sqrt(1.0 / n) if k == 0 else np.sqrt(2.0 / n)
        for i in range(n):
            mat[k, i] = scale * np.cos((2 * i + 1) * k * factor)

    _DCT_CACHE[n] = mat
    return mat


def _dct_2d(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    mat = _dct_matrix(n)
    return mat @ values @ mat.T


def compute_phash(
    image_path: str,
    hash_size: int = 8,
    highfreq_factor: int = 4,
) -> Optional[int]:
    """Compute perceptual hash for an image path."""
    size = hash_size * highfreq_factor
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            pixels = np.asarray(img, dtype=np.float32)
    except Exception:
        return None

    dct = _dct_2d(pixels)
    dct_low = dct[:hash_size, :hash_size]
    flat = dct_low.flatten()
    if flat.size <= 1:
        return None

    median = float(np.median(flat[1:]))
    bits = flat > median
    hash_value = 0
    for bit in bits:
        hash_value = (hash_value << 1) | int(bool(bit))
    return int(hash_value)


def get_phash(image_path: str) -> Optional[int]:
    """Return cached perceptual hash for image path."""
    with _CACHE_LOCK:
        if image_path in _PHASH_CACHE:
            return _PHASH_CACHE[image_path]

    phash = compute_phash(image_path)
    with _CACHE_LOCK:
        _PHASH_CACHE[image_path] = phash
    return phash


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two hashes."""
    xor = a ^ b
    try:
        return xor.bit_count()
    except AttributeError:
        return bin(xor).count("1")


def _cluster_indices_by_hamming(
    hashes: List[Optional[int]],
    max_distance: int,
) -> List[List[int]]:
    if not hashes:
        return []
    uf = _UnionFind(len(hashes))
    for i, hash_i in enumerate(hashes):
        if hash_i is None:
            continue
        for j in range(i + 1, len(hashes)):
            hash_j = hashes[j]
            if hash_j is None:
                continue
            if hamming_distance(hash_i, hash_j) <= max_distance:
                uf.union(i, j)
    return uf.clusters()


def group_paths_by_phash(
    image_paths: List[str],
    max_distance: int = DEFAULT_PHASH_DISTANCE,
    max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
) -> List[List[str]]:
    """Group images by pHash Hamming distance."""
    paths = list(image_paths)
    n = len(paths)
    if n == 0:
        return []
    if n == 1:
        return [paths]
    if max_group_size and n > max_group_size:
        return [[path] for path in paths]

    hashes = [get_phash(path) for path in paths]
    cluster_indices = _cluster_indices_by_hamming(hashes, max_distance)
    return [[paths[i] for i in indices] for indices in cluster_indices]
