from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pandas as pd

# Initialize SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_features(texts, chunk_size=100):
    """
    Extracts features in chunks and shows overall progress.
    """
    # Convert texts to list if it's a pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    features = []
    # Show progress for the entire feature extraction
    for i in tqdm(range(0, len(texts), chunk_size), desc="Extracting features", total=len(texts)//chunk_size):
        chunk = texts[i:i + chunk_size]
        chunk_features = model.encode(chunk, convert_to_tensor=False)
        features.extend(chunk_features)
    return np.vstack(features)
