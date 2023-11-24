import torch
from transformers import RobertaTokenizer, RobertaModel
import numpy as np
from tqdm import tqdm

# Check for MPS (Apple Silicon GPU) support
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(device)


def extract_features(texts):
    """
    Extracts RoBERTa embeddings for a list of texts.
    ...
    """
    features = []
    for text in tqdm(texts, desc="Extracting features", unit="text"):
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**inputs)
        feature_vec = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        features.append(feature_vec)
    # Convert list of arrays into a 2D numpy array
    return np.vstack(features)

