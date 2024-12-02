import numpy as np
from joblib import load
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


# Load models and vectorizers
tfidf_vectoriser = load("models/tfidf_vectorizer.joblib")
nmf_model = load("models/nmf_model.joblib")
glove_model = load("models/glove_model.joblib")
scaler_tfidf = load("models/scaler_tfidf.joblib")
scaler_glove = load("models/scaler_glove.joblib")
scaler_nmf = load("models/scaler_nmf.joblib")


def compute_glove_embedding(tokens: list[str], glove_model) -> np.ndarray:
    """
    Compute the average GloVe embedding for the given tokens.

    Args:
        tokens (list[str]): List of tokens extracted from the text.
        glove_model: Pre-trained GloVe model.

    Returns:
        np.ndarray: Average GloVe embedding vector for the input tokens.
    """
    embeddings = [glove_model[token] for token in tokens if token in glove_model]

    # print(f"Tokens: {tokens}")
    # print(f"Embeddings: {embeddings}")

    if embeddings:  # Ensure there are valid embeddings
        return np.mean(embeddings, axis=0)
    else:
        # Handle cases where no tokens match the GloVe vocabulary
        return np.zeros(glove_model.vector_size)


def get_nmf_features(tfidf_features: np.ndarray) -> np.ndarray:
    """
    Extract NMF features using the pre-trained NMF model.

    Args:
        tfidf_features (np.ndarray): TF-IDF feature array.

    Returns:
        np.ndarray: Flattened NMF feature array.
    """
    # Get topic distribution using the NMF model
    nmf_features = nmf_model.transform(tfidf_features)

    return nmf_features.flatten()


def extract_features(
    reviews: str, tfidf_weight: float = 0.4, glove_weight: float = 0.3, nmf_weight: float = 0.3
) -> np.ndarray:
    """
    Extract features from preprocessed text using TF-IDF, GloVe, and NMF.

    Args:
        reviews (str): Preprocessed text for feature extraction.
        tfidf_weight (float, optional): Weight for TF-IDF features. Defaults to 0.4.
        glove_weight (float, optional): Weight for GloVe features. Defaults to 0.3.
        nmf_weight (float, optional): Weight for NMF features. Defaults to 0.3.

    Returns:
        np.ndarray: Combined feature vector consisting of scaled TF-IDF, GloVe, and NMF features.
    """
    # TF-IDF features
    tfidf_features = tfidf_vectoriser.transform([reviews]).toarray()

    # GloVe features
    tokens = reviews.split()  # Tokenize preprocessed text
    glove_features = compute_glove_embedding(tokens, glove_model).reshape(1, -1)

    # NMF features
    nmf_features = get_nmf_features(tfidf_features).reshape(1, -1)  # Ensure NMF features are 2D

    tfidf_scaled = scaler_tfidf.transform(tfidf_features)
    glove_scaled = scaler_glove.transform(glove_features)
    nmf_scaled = scaler_nmf.transform(nmf_features)

    combined_features = np.hstack(
        [tfidf_scaled, glove_scaled, nmf_scaled]
    )

    return combined_features
