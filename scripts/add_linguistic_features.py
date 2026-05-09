import pandas as pd
import re
import os

DATA_DIR = "/Users/shubhampatthe/Downloads/Data/datasets/finbert"

HEDGE_WORDS = ['approximately', 'perhaps', 'might', 'could', 'possibly', 
               'uncertain', 'believe', 'expect', 'roughly', 'around', 
               'may', 'seems', 'appears', 'suggest', 'likely']

COMPLEXITY_WORDS = ['notwithstanding', 'pursuant', 'accordingly', 'thereafter', 
                    'hereinafter', 'whereby', 'aforementioned']

def extract_features(text):
    if not isinstance(text, str):
        text = str(text)
    words = text.lower().split()
    total = len(words) + 1e-9
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    
    return {
        'hedge_ratio':      sum(w in HEDGE_WORDS for w in words) / total,
        'complexity_ratio': sum(w in COMPLEXITY_WORDS for w in words) / total,
        'avg_sentence_len': total / (len(sentences) + 1e-9),
        'unique_word_ratio': len(set(words)) / total,
        'question_count':   text.count('?'),
        'number_ratio':     sum(bool(re.search(r'\d', w)) for w in words) / total,
    }

def main():
    print("Starting linguistic feature extraction...")
    for fname in ['train_dataset.csv', 'val_dataset.csv', 'test_dataset.csv']:
        file_path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(file_path):
            print(f"Skipping {fname} - not found.")
            continue
            
        print(f"Processing {fname}...")
        df = pd.read_csv(file_path)
        
        # Apply feature extraction
        features_df = df['Text'].apply(extract_features).apply(pd.Series)
        
        # Combine original df with new features
        enhanced_df = pd.concat([df, features_df], axis=1)
        
        # Save back to CSV
        enhanced_df.to_csv(file_path, index=False)
        print(f"✅ Updated {fname} with linguistic features.")

if __name__ == "__main__":
    main()
