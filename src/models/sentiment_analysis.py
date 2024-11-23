import joblib
from utils.preprocessor.preprocess import advance_text_cleaning
from utils.features_extraction.extract_features import extract_features


model = joblib.load("../models/logistic_sentiment_model.joblib")


def analyse_sentiments(reviews):


    preprocessed_review = advance_text_cleaning(reviews)

    features = extract_features(preprocessed_review)

    sentiment = model.predict(features)
    
    return sentiment[0]  # Return the prediction (usually a single value for one input)
   

