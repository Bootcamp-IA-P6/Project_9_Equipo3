import nlpaug.augmenter.word as naw
import pandas as pd
from tqdm import tqdm
import nltk

def download_nltk_resources():
    """
    Ensure all required NLTK resources are downloaded.
    """
    resources = ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}')
        except LookupError:
            nltk.download(res)

def augment_text(df, target_cols, multiplier=1):
    """
    Augment minority samples using Synonym replacement.
    """
    # 1. Ensure resources are available
    download_nltk_resources()
    
    # 2. Initialize augmenter
    # Using 'wordnet' is standard, but initialization can be slow
    aug = naw.SynonymAug(aug_src='wordnet')
    
    augmented_records = []
    
    # 3. Filter minority samples (where at least one toxic label is 1)
    minority_df = df[df[target_cols].sum(axis=1) > 0].copy()
    
    if minority_df.empty:
        print("No minority samples found to augment.")
        return df

    print(f"Starting augmentation for {len(minority_df)} samples...")
    
    # 4. Augmentation loop
    for _ in range(multiplier):
        for _, row in tqdm(minority_df.iterrows(), total=len(minority_df), desc="Augmenting"):
            try:
                # Synonym replacement
                text = row['Text']
                if not text or pd.isna(text):
                    continue
                    
                new_text_list = aug.augment(text)
                new_text = new_text_list[0] if isinstance(new_text_list, list) else new_text_list
                
                new_row = row.copy()
                new_row['Text'] = new_text
                augmented_records.append(new_row)
            except Exception as e:
                # Skip if augmentation fails for a specific row
                continue
            
    augmented_df = pd.concat([df, pd.DataFrame(augmented_records)], ignore_index=True)
    return augmented_df