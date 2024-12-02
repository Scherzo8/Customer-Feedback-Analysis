import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from utils.preprocessor.preprocess import advance_text_cleaning
from utils.features_extraction.extract_features import extract_features

# Load pre-trained models and tokenizer
logistic_model = joblib.load("models/logistic_sentiment_model.joblib")
bert_model = BertForSequenceClassification.from_pretrained("models/sentiment_bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("models/sentiment_bert_model")


def analyse_sentiments(reviews: str, model_type: str = "logistic") -> str:
    """
    Analyzes the sentiment of a given review using the specified model.

    Args:
        reviews (str): The review text to be analyzed.
        model_type (str): The model to use for sentiment analysis ('logistic' or 'bert').

    Returns:
        str: The predicted sentiment label ('Positive', 'Negative', or 'Neutral').
    """
    # Preprocess the review text
    preprocessed_review = advance_text_cleaning(reviews)

    if model_type == "logistic":
        # Logistic Regression Model Inference
        features = extract_features(preprocessed_review)  # Extract features for logistic regression
        sentiment = logistic_model.predict(features)

        # Return sentiment as a string
        if sentiment[0] == "Positive":
            return "Positive"
        elif sentiment[0] == "Neutral":
            return "Neutral"
        else:
            return "Negative"

    elif model_type == "bert":
        # BERT Model Inference
        inputs = bert_tokenizer(
            preprocessed_review,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        # Map BERT output to sentiment labels
        if predicted_class == 2:
            return "Positive"
        elif predicted_class == 1:
            return "Neutral"
        elif predicted_class == 0:
            return "Negative"

    else:
        raise ValueError("Invalid model type. Choose 'logistic' or 'bert'.")
