"""NumPy-based mathematical utilities to replace scikit-learn dependencies."""

import numpy as np
from typing import Optional


class PCA:
    """Principal Component Analysis implementation using NumPy.

    Replacement for sklearn.decomposition.PCA to reduce dependencies.
    """

    def __init__(self, n_components: Optional[int] = None):
        """Initialize PCA.

        Args:
            n_components: Number of components to keep. If None, keep all.
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Fit PCA model.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top n_components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        # Store components
        self.components_ = eigenvectors.T
        self.explained_variance_ = eigenvalues

        # Calculate explained variance ratio
        total_variance = np.sum(np.var(X_centered, axis=0))
        self.explained_variance_ratio_ = eigenvalues / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction.

        Args:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA must be fitted before transform")

        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA model and transform data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)


def cosine_similarity(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute cosine similarity between samples.

    Replacement for sklearn.metrics.pairwise.cosine_similarity.

    Args:
        X: Array of shape (n_samples_X, n_features)
        Y: Array of shape (n_samples_Y, n_features). If None, use X.

    Returns:
        Cosine similarity matrix of shape (n_samples_X, n_samples_Y)
    """
    if Y is None:
        Y = X

    # Normalize vectors
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_normalized = Y / np.linalg.norm(Y, axis=1, keepdims=True)

    # Compute dot product (cosine similarity)
    return np.dot(X_normalized, Y_normalized.T)


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.

    Optimized version for computing similarity between all pairs.

    Args:
        embeddings: Array of shape (n_samples, n_features)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    # Compute similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)

    # Clip to handle numerical errors
    return np.clip(similarity_matrix, -1.0, 1.0)
