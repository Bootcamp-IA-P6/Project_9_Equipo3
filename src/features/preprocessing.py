import re
import string
import pandas as pd
from textblob import TextBlob
import numpy as np

def clean_text(text: str):
    """
    Apply regex and basic string operations to clean the input text.
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove user mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_advanced_features(df: pd.DataFrame):
    """
    Extract metadata and sentiment features to distinguish intent from topic.
    """
    # 1. Sentiment Analysis (Polarity: -1 to 1, Subjectivity: 0 to 1)
    # Toxic comments are often highly negative and subjective.
    sentiments = df['Text'].apply(lambda x: TextBlob(x).sentiment)
    df['polarity'] = [s.polarity for s in sentiments]
    df['subjectivity'] = [s.subjectivity for s in sentiments]
    
    # 2. Stylometric Features
    df['caps_ratio'] = df['Text'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['excl_count'] = df['Text'].apply(lambda x: x.count('!'))
    df['quest_count'] = df['Text'].apply(lambda x: x.count('?'))
    df['text_len'] = df['Text'].apply(len)
    df['avg_word_len'] = df['Text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    
    return df[['polarity', 'subjectivity', 'caps_ratio', 'excl_count', 'quest_count', 'text_len', 'avg_word_len']]

if __name__ == "__main__":
    sample = "Check this out! https://example.com @user Great video!!!"
    print(f"Original: {sample}")
    print(f"Cleaned: {clean_text(sample)}")