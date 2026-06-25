import numpy as np


def l1_distance(v1, v2):
    # Sum of absolute differences.
    return np.sum(np.abs(v1 - v2))


def l2_distance(v1, v2):
    # Euclidean distance between two vectors.
    return np.linalg.norm(v1 - v2)


def cosine_similarity(v1, v2):
    # Compare the direction of two vectors.
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero for empty/black vectors.
    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)
