import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = "/Users/shubhampatthe/Downloads/Data"
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
FINBERT_DIR = os.path.join(DATASETS_DIR, "finbert")

# Define the fraud and non-fraud companies based on your feedback
FRAUD_COMPANIES = [
    "General Electric",
    "MiMedx",
    "Satyam",
    "Under Armour",
    "Valeant_Pharmaceuticals",
    "Weatherfordinternationals"
]

NON_FRAUD_COMPANIES = [
    "3M",
    "Integra LifeSciences",
    "TATA CONSULTANCY SERVICES",
    "Johnson and Johnson",
    "Nike",
    "Schlumberger"
]

# Exact fraud periods based on SEC Enforcement actions
FRAUD_PERIODS = {
    "General Electric": (2015, 1, 2017, 4),
    "MiMedx": (2013, 1, 2017, 4),
    "Satyam": (2003, 1, 2008, 3),
    "Under Armour": (2015, 3, 2016, 4),
    "Valeant_Pharmaceuticals": (2014, 1, 2015, 4),
    "Weatherfordinternationals": (2007, 1, 2012, 4)
}

def get_label(company, year, quarter):
    """
    Returns 1 for fraud period, 0 for pre-fraud and non-fraud,
    and None for post-fraud (to exclude them to ensure clean mapping).
    """
    if company in NON_FRAUD_COMPANIES:
        return 0
    elif company in FRAUD_COMPANIES:
        start_year, start_q, end_year, end_q = FRAUD_PERIODS[company]
        
        # Check if the current period is strictly before the fraud period (Pre-fraud)
        if year < start_year or (year == start_year and quarter < start_q):
            return 0
        # Check if the current period is within the fraud period
        elif (year > start_year or (year == start_year and quarter >= start_q)) and \
             (year < end_year or (year == end_year and quarter <= end_q)):
            return 1
        else:
            # Post-fraud period. We return None to filter these out.
            return None
    return None

def chunk_text(text, max_words=350):
    """Chunks text into smaller pieces of roughly max_words (safe for FinBERT's 512 token limit)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def main():
    records = []
    
    for company_dir in os.listdir(TRANSCRIPTS_DIR):
        full_dir_path = os.path.join(TRANSCRIPTS_DIR, company_dir)
        if not os.path.isdir(full_dir_path):
            continue
            
        company_name = company_dir
        if company_name not in FRAUD_COMPANIES and company_name not in NON_FRAUD_COMPANIES:
            continue
            
        for filename in os.listdir(full_dir_path):
            if not filename.endswith(".txt"):
                continue
                
            match = re.search(r'Q_(\d)_?\s*(\d{4})', filename)
            if not match:
                continue
                
            quarter = int(match.group(1))
            year = int(match.group(2))
            
            label = get_label(company_name, year, quarter)
            if label is None:
                continue # Skip post-fraud periods
                
            file_path = os.path.join(full_dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
                
            # Chunk the text
            chunks = chunk_text(text, max_words=350)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Filter out very short chunks (e.g., greetings, sign-offs)
                if len(chunk.split()) < 30:
                    continue
                    
                records.append({
                    "Company": company_name,
                    "Year": year,
                    "Quarter": quarter,
                    "Chunk_ID": chunk_idx,
                    "Text": chunk,
                    "Label": label
                })
                
    if not records:
        print("No valid records found. Please check paths and parsing logic.")
        return

    df = pd.DataFrame(records)
    output_path = os.path.join(DATASETS_DIR, "fraud_dataset_chunked.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset created successfully with {len(df)} chunks.")
    print(f"✅ Saved to: {output_path}")
    
    print("\n--- Class Distribution (Overall) ---")
    print(df['Label'].value_counts(normalize=True))
    print(df['Label'].value_counts())
    
    # Stratified Train/Val/Test Split (Document-Level to Prevent Leakage)
    print("\n--- Performing Stratified Sampling (Document-Level) ---")
    
    # Create a unique Document ID for splitting
    df['Doc_ID'] = df['Company'] + '_' + df['Year'].astype(str) + '_' + df['Quarter'].astype(str)
    
    # Get unique documents and their labels
    docs_df = df[['Doc_ID', 'Label']].drop_duplicates()
    
    # Split documents (80% Train, 10% Val, 10% Test)
    train_docs, temp_docs = train_test_split(
        docs_df, test_size=0.2, stratify=docs_df['Label'], random_state=42
    )
    val_docs, test_docs = train_test_split(
        temp_docs, test_size=0.5, stratify=temp_docs['Label'], random_state=42
    )
    
    # Filter the original chunked dataframe based on the document splits
    train_df = df[df['Doc_ID'].isin(train_docs['Doc_ID'])].drop(columns=['Doc_ID'])
    val_df = df[df['Doc_ID'].isin(val_docs['Doc_ID'])].drop(columns=['Doc_ID'])
    test_df = df[df['Doc_ID'].isin(test_docs['Doc_ID'])].drop(columns=['Doc_ID'])
    
    # Drop Doc_ID from the main df before saving if needed, but it's fine.
    df = df.drop(columns=['Doc_ID'])
    
    train_df.to_csv(os.path.join(FINBERT_DIR, "train_dataset.csv"), index=False)
    val_df.to_csv(os.path.join(FINBERT_DIR, "val_dataset.csv"), index=False)
    test_df.to_csv(os.path.join(FINBERT_DIR, "test_dataset.csv"), index=False)
    
    print(f"✅ Created stratified splits:")
    print(f"  - Train: {len(train_df)} chunks")
    print(f"  - Val:   {len(val_df)} chunks")
    print(f"  - Test:  {len(test_df)} chunks")
    
    # Ensure no data leakage. Although chunks from the same document might end up in different splits.
    # A Grouped Stratified split by 'Company' and 'Year'/'Quarter' would be better to avoid data leakage 
    # (i.e. different chunks of the SAME transcript in train AND test). 
    # But for now, basic stratification is applied.

if __name__ == "__main__":
    main()
