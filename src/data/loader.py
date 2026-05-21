import pandas as pd

def load_processed_data(filepath: str):
    """
    Load the YouTube toxic comments dataset and perform basic cleaning.
    """
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    df = df.drop(columns=['CommentId', 'VideoId'])
    
    # Convert string boolean (TRUE/FALSE) to integers (1/0)
    # The columns from 'IsToxic' onwards are labels
    label_cols = df.columns.drop('Text')
    for col in label_cols:
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0, True: 1, False: 0})
        
    return df, label_cols

if __name__ == "__main__":
    # Test loading
    data, labels = load_processed_data("data/raw/youtoxic_english_1000.csv")
    print(f"Data Loaded. Shape: {data.shape}")
    print(f"Labels: {list(labels)}")