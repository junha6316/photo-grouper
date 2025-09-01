#!/usr/bin/env python3
"""
Script to calculate VGG16 embedding cosine similarity between two images.
Usage: python compare_images.py <image1_path> <image2_path>
"""

import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
from core.embedder import ImageEmbedder


def calculate_cosine_similarity(image_path1: str, image_path2: str) -> float:
    """
    Calculate cosine similarity between two images using VGG16 embeddings.

    Args:
        image_path1: Path to first image
        image_path2: Path to second image

    Returns:
        Cosine similarity score (0-1, where 1 is identical)
    """
    # Initialize embedder
    embedder = ImageEmbedder()

    # Get raw embeddings (without PCA)
    embedding1 = embedder._get_raw_embedding(image_path1)
    embedding2 = embedder._get_raw_embedding(image_path2)

    # Calculate cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]

    return float(similarity)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_images.py <image1_path> <image2_path>")
        sys.exit(1)

    image_path1 = sys.argv[1]
    image_path2 = sys.argv[2]

    # Check if files exist
    if not os.path.exists(image_path1):
        print(f"Error: Image file not found: {image_path1}")
        sys.exit(1)

    if not os.path.exists(image_path2):
        print(f"Error: Image file not found: {image_path2}")
        sys.exit(1)

    # Calculate similarity
    try:
        similarity = calculate_cosine_similarity(image_path1, image_path2)
        print(f"Cosine similarity: {similarity:.4f}")

        # Provide interpretation
        if similarity > 0.9:
            print("Interpretation: Very similar images")
        elif similarity > 0.8:
            print("Interpretation: Similar images")
        elif similarity > 0.6:
            print("Interpretation: Somewhat similar images")
        else:
            print("Interpretation: Different images")

    except Exception as e:
        print(f"Error calculating similarity: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


# 정렬 알고리즘에 문제가 있네