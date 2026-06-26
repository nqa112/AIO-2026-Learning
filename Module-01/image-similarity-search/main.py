import os

import cv2
import numpy as np

from metrics import cosine_similarity, l1_distance, l2_distance


IMAGE_SIZE = (64, 64)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def process_image(image, size=IMAGE_SIZE):
    # Read image in grayscale to reduce the number of channels
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot load image: {image}")
    else:
        img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize every image to the same size so vectors have equal length.
    img_resized = cv2.resize(img, size)

    # Convert the 2D image matrix into a 1D vector
    vector = img_resized.flatten()
    # Scale pixels to [0, 1].
    vector_normalized = vector / 255.0

    return vector_normalized


def calculate_score(query_vector, image_vector, metric):
    # Smaller is better.
    if metric == "L1":
        return l1_distance(query_vector, image_vector)

    # Smaller is better.
    if metric == "L2":
        return l2_distance(query_vector, image_vector)

    # Larger is better.
    if metric == "Cosine":
        return cosine_similarity(query_vector, image_vector)

    raise ValueError(f"Unknown metric: {metric}")


def load_dataset_vectors(dataset_dir):
    # Process dataset images once and keep their vectors for searching.
    dataset_vectors = []

    for filename in os.listdir(dataset_dir):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(dataset_dir, filename)
        image_vector = process_image(image_path)

        dataset_vectors.append(
            {
                "filename": filename,
                "path": image_path,
                "vector": image_vector,
            }
        )

    return dataset_vectors


def find_similar_images(query_image, dataset_vectors, metric="L2", top_k=3):
    # Convert the query image once, then compare it with all dataset images.
    query_vector = process_image(query_image)
    results = []

    for item in dataset_vectors:
        image_vector = item["vector"]
        score = calculate_score(query_vector, image_vector, metric)

        results.append(
            {
                "filename": item["filename"],
                "path": item["path"],
                "score": score,
            }
        )

    # Set order for sorting score based on chosen metric
    reverse = metric == "Cosine"
    results = sorted(results, key=lambda item: item["score"], reverse=reverse)

    return results[:top_k]
