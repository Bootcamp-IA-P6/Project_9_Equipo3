import random
import re
from typing import List

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}


def _get_synonyms(word: str) -> List[str]:
    """Look up other words that mean roughly the same thing, using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ").replace("-", " ").lower()
            if name != word.lower() and name.isalpha():
                synonyms.add(name)
    return list(synonyms)


class TextAugmenter:
    """Builds new training sentences by swapping some words for synonyms."""

    def __init__(self, replacement_prob: float = 0.15, max_replacements: int = 3):
        self.replacement_prob = replacement_prob
        self.max_replacements = max_replacements

    def augment(self, text: str) -> str:
        """Take one sentence and return a slightly reworded copy of it."""
        if not text or not isinstance(text, str):
            return text

        words = word_tokenize(text)
        if len(words) < 3:
            return text

        augmented_words = words.copy()
        candidate_indices = [
            i for i, w in enumerate(words)
            if w.lower() not in _STOP_WORDS and w.isalpha()
        ]

        if not candidate_indices:
            return text

        # Reword just a few words, not the whole sentence
        n_replacements = min(self.max_replacements, max(1, int(len(candidate_indices) * self.replacement_prob)))
        replace_indices = random.sample(candidate_indices, min(len(candidate_indices), n_replacements))

        replaced_any = False
        for idx in replace_indices:
            word = words[idx]
            synonyms = _get_synonyms(word)
            if synonyms:
                synonym = random.choice(synonyms)
                augmented_words[idx] = synonym
                replaced_any = True

        if not replaced_any:
            # No synonyms were found, so just shuffle two words around instead
            if len(augmented_words) >= 2:
                idx1, idx2 = random.sample(range(len(augmented_words)), 2)
                augmented_words[idx1], augmented_words[idx2] = augmented_words[idx2], augmented_words[idx1]

        # Glue the words back into a sentence
        augmented_text = " ".join(augmented_words)
        # Drop the stray space the join left before punctuation
        augmented_text = re.sub(r"\s+([.,!?;'])", r"\1", augmented_text)
        return augmented_text


def augment_dataset(X, y, label_to_augment: int = 1, multiplier: float = 0.4, random_state: int = 42) -> tuple:
    """
    Grow one class of the dataset by rewording its existing examples.
    Handy when the toxic class is too small; only that class gets new rows.
    
    Args:
        X: The text samples.
        y: The matching labels.
        label_to_augment: Which class to grow (1 = toxic by default).
        multiplier: How much to grow it. 0.4 means "add 40% more toxic rows".
        random_state: Fixed seed so the same rows come out on every run.
        
    Returns:
        (X, y) with the new synthetic rows added at the end.
    """
    import numpy as np
    import pandas as pd

    random.seed(random_state)
    np.random.seed(random_state)

    X_series = pd.Series(X)
    y_series = pd.Series(y)

    target_mask = y_series == label_to_augment
    X_target = X_series[target_mask]
    
    n_samples_to_augment = int(len(X_target) * multiplier)
    if n_samples_to_augment == 0:
        return X_series, y_series

    # Randomly pick which toxic rows to reword (the same row can be picked twice)
    indices_to_augment = np.random.choice(X_target.index, size=n_samples_to_augment, replace=True)
    
    augmenter = TextAugmenter()
    synthetic_texts = []
    
    for idx in indices_to_augment:
        orig_text = X_target.loc[idx]
        synthetic_texts.append(augmenter.augment(orig_text))

    synthetic_labels = [label_to_augment] * len(synthetic_texts)

    # Tack the new synthetic rows onto the end of the dataset
    X_augmented = pd.concat([X_series, pd.Series(synthetic_texts)], ignore_index=True)
    y_augmented = pd.concat([y_series, pd.Series(synthetic_labels)], ignore_index=True)

    return X_augmented, y_augmented
