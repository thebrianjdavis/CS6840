from sentence_transformers import SentenceTransformer
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch


# Check for MPS (Apple Silicon GPU) support and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Initialize SBERT model and move it to the specified device
model_sbert = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Initialize the RoBERTa tokenizer and model, and move the model to the specified device
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model_roberta = RobertaModel.from_pretrained('roberta-base').to(device)


def extract_features_sbert(texts, chunk_size=100):
    """
    Extracts features in chunks and shows overall progress.

    This function processes texts in chunks and uses the SBERT model to extract
    features. If Apple Silicon GPU is available, the model utilizes it for faster processing.

    @param texts A list or pandas Series of text strings for feature extraction.
    @param chunk_size The number of texts to process in each chunk.
    @return A 2D numpy array of extracted features.
    """
    # Convert texts to list if it's a pandas Series
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    features = []
    # Show progress for the entire feature extraction
    for i in tqdm(range(0, len(texts), chunk_size), desc="Extracting features", total=len(texts) // chunk_size):
        chunk = texts[i:i + chunk_size]
        # Encode the chunk of texts, converting them to tensors on the specified device
        chunk_features = model_sbert.encode(chunk, convert_to_tensor=True).to(device)
        # Move the features back to CPU and convert to numpy array
        chunk_features = chunk_features.cpu().numpy()
        features.extend(chunk_features)
    return np.vstack(features)


def extract_features_roberta(texts):
    """
    Extracts RoBERTa embeddings for a list of texts.

    This function processes each text in the input list, tokenizing the text and then
    passing it through the RoBERTa model to extract feature embeddings. The embeddings
    are averaged over all tokens to create a single feature vector per text.

    @param texts List of text strings for which features are to be extracted.
    @return A 2D numpy array where each row corresponds to the feature vector of a text.
    """
    features = []
    for text in tqdm(texts, desc="Extracting features", unit="text"):
        # Tokenize the text and move to the appropriate device
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        # Get the model output
        outputs = model_roberta(**inputs)
        # Extract the feature vector and move it to CPU
        feature_vec = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        features.append(feature_vec)
    # Convert list of arrays into a 2D numpy array
    return np.vstack(features)
