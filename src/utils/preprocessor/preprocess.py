import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Initialise lemmitiser and stopwords
lemmitiser = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def advance_text_cleaning(text: str) -> str:
    """
    Cleans the text using techniques: lowercasing, removing special characters, 
    removing stopwords, and lemmatization.
    
    Args:  
        text (str): Input text to be cleaned.
        
    Returns:
        str: Cleaned and preprocessed text
    """

    if not isinstance(text, str):  return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    
    # Remove special characters, numbers, and retain only alphabets and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowecase
    text = text.lower()

    # Tokenise text
    words = word_tokenize(text=text)

    # Remove stopwords and lemmitise words
    words = [
        lemmitiser.lemmatize(word=word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)