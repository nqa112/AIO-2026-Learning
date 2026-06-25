from pathlib import Path

import streamlit as st
from PIL import Image

from main import find_similar_images, load_dataset_vectors


DATA_DIR = Path("Data")
DATASET_DIR = DATA_DIR / "images_folder"
DEFAULT_QUERY_IMAGE = DATA_DIR / "query.jpg"


# Cache dataset vectors to avoid Streamlit rerun
@st.cache_data
def get_dataset_vectors(dataset_dir):
    return load_dataset_vectors(dataset_dir)


st.title("Image Similarity Search")

# If user does not upload any image, use Data/query.jpg
uploaded_file = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png"],
)

# Choose metrics and how many images to be searched
metric = st.selectbox("Choose metric", ["L2", "L1", "Cosine"])
top_k = st.slider("Number of results", min_value=1, max_value=5, value=3)

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")
else:
    query_image = Image.open(DEFAULT_QUERY_IMAGE).convert("RGB")

st.subheader("Query Image")
st.image(query_image, width=250)

if st.button("Search"):
    dataset_vectors = get_dataset_vectors(str(DATASET_DIR))
    results = find_similar_images(query_image, dataset_vectors, metric, top_k)

    st.subheader("Similar Images")

    for result in results:
        st.image(result["path"], width=220)
        st.write(f"File: {result['filename']}")
        st.write(f"Score: {result['score']:.4f}")
