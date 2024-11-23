import numpy as np
from joblib import load
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


# Load models and vectorizers
tfidf_vectoriser = load("../models/tfidf_vectorizer.joblib")
nmf_model = load("../models/nmf_model.joblib")
glove_model = load("../models/glove_model.joblib")
scaler_tfidf = load("../models/scaler_tfidf.joblib")
scaler_glove = load("../models/scaler_glove.joblib")
scaler_nmf = load("../models/scaler_nmf.joblib")


def compute_glove_embedding(tokens, glove_model):
    """
    Compute the average GloVe embedding for the given tokens.
    """
    embeddings = [glove_model[token] if token in glove_model else np.zeros(300) for token in tokens]

    if embeddings:  return np.mean(embeddings, axis=0)

    else:  return np.zeros(300)  # Return a zero vector if no tokens are found


def get_nmf_features(tfidf_features):
    """
    Extract NMF features using the pre-trained NMF model.
    """
    # Get topic distribution using the NMF model
    nmf_features = nmf_model.transform(tfidf_features)

    return nmf_features.flatten()


def extract_features(reviews: str, tfidf_weight=0.4, glove_weight=0.3, lda_weight=0.3):
    """
    Extract features from preprocessed text using TF-IDF, GloVe, and LDA.
    """

    # TF-IDF features
    tfidf_features = tfidf_vectoriser.transform([reviews]).toarray()

    # GloVe features
    tokens = reviews.split()  # Tokenize preprocessed text
    glove_features = compute_glove_embedding(tokens, glove_model).reshape(1, -1)

    # LDA features
    nmf_features = get_nmf_features(tfidf_features).reshape(1, -1)  # Ensure NMF features are 2D

    tfidf_scaled = scaler_tfidf.transform(tfidf_features)
    glove_scaled = scaler_glove.transform(glove_features)
    nmf_scaled = scaler_nmf.transform(nmf_features)

    # Weighted sum of features
    combined_features = np.hstack(
        [tfidf_scaled, glove_scaled, nmf_scaled]
    )

    return combined_features
