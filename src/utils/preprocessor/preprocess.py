import re
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.util import mark_negation
from nltk.stem import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary


# Initialize resources
lemmatizer = WordNetLemmatizer()

# Load stopwords and exclude negation words
stop_words = set(stopwords.words("english"))


def advance_text_cleaning(text: str) -> str:
    """
    Cleans the text using techniques: lowercasing, expanding contractions, 
    removing special characters, handling negations, removing stopwords, 
    lemmatization, and optional emoji removal.
    
    Args:  
        text (str): Input text to be cleaned.
        
    Returns:
        str: Cleaned and preprocessed text
    """

    if not isinstance(text, str):  
        return ""

    # Expand contractions (e.g., "can't" → "cannot")
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)  # Removes emojis and other symbols
    
    # Remove special characters and numbers, retain only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase
    text = text.lower()

    # Tokenize text and handle negations
    words = word_tokenize(text)
    # words = mark_negation(words)  # E.g., ["not", "good"] → ["not_good"]

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Convert tokens back to string
    return " ".join(words)