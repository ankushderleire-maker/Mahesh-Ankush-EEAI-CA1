import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import Config

def get_tfidf_embd(df: pd.DataFrame):
    # Combine ticket summary and interaction content for feature extraction
    combined_text = df[Config.TICKET_SUMMARY].astype(str) + " " + df[Config.INTERACTION_CONTENT].astype(str)
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(combined_text).toarray()
    
    return X