import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR    = "/Users/shubhampatthe/Downloads/Data"
TRANSCRIPTS_DIR = os.path.join(DATA_DIR, "transcripts")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
OUTPUT_DIR  = os.path.join(DATASETS_DIR, "longformer")
CHUNK_WORDS = 1500      # Larger chunks for Longformer's 4096-token window

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

FRAUD_PERIODS = {
    "General Electric":        (2015, 1, 2017, 4),
    "MiMedx":                  (2013, 1, 2017, 4),
    "Satyam":                  (2003, 1, 2008, 3),
    "Under Armour":            (2015, 3, 2016, 4),
    "Valeant_Pharmaceuticals": (2014, 1, 2015, 4),
    "Weatherfordinternationals":(2007, 1, 2012, 4)
}

def get_label(company, year, quarter):
    if company in NON_FRAUD_COMPANIES:
        return 0
    if company in FRAUD_COMPANIES:
        sy, sq, ey, eq = FRAUD_PERIODS[company]
        is_before = (year < sy) or (year == sy and quarter < sq)
        is_during = ((year > sy) or (year == sy and quarter >= sq)) and \
                    ((year < ey) or (year == ey and quarter <= eq))
        if is_before:
            return 0
        if is_during:
            return 1
        return None   # post-fraud — exclude
    return None

def chunk_text(text, max_words=CHUNK_WORDS):
    """Split text into max_words-word chunks with a 100-word overlap for context continuity."""
    words = text.split()
    step  = max_words - 100     # 100-word sliding overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks

def extract_linguistic_features(text):
    """Identical feature set as the FinBERT model for a fair comparison."""
    HEDGE_WORDS = ['approximately', 'perhaps', 'might', 'could', 'possibly',
                   'uncertain', 'believe', 'expect', 'roughly', 'around',
                   'may', 'seems', 'appears', 'suggest', 'likely']
    COMPLEXITY_WORDS = ['notwithstanding', 'pursuant', 'accordingly', 'thereafter',
                        'hereinafter', 'whereby', 'aforementioned']
    words    = text.lower().split()
    total    = len(words) + 1e-9
    sentences = [s for s in re.split(r'[.!?]', text) if len(s.strip()) > 0]
    return {
        'hedge_ratio':      sum(w in HEDGE_WORDS for w in words) / total,
        'complexity_ratio': sum(w in COMPLEXITY_WORDS for w in words) / total,
        'avg_sentence_len': total / (len(sentences) + 1e-9),
        'unique_word_ratio': len(set(words)) / total,
        'question_count':   text.count('?'),
        'number_ratio':     sum(bool(re.search(r'\d', w)) for w in words) / total,
    }

def main():
    records = []

    for company_dir in sorted(os.listdir(TRANSCRIPTS_DIR)):
        full_dir_path = os.path.join(TRANSCRIPTS_DIR, company_dir)
        if not os.path.isdir(full_dir_path):
            continue
        company = company_dir
        if company not in FRAUD_COMPANIES and company not in NON_FRAUD_COMPANIES:
            continue

        for filename in sorted(os.listdir(full_dir_path)):
            if not filename.endswith(".txt"):
                continue
            match = re.search(r'Q_(\d)_?\s*(\d{4})', filename)
            if not match:
                continue
            quarter = int(match.group(1))
            year    = int(match.group(2))

            label = get_label(company, year, quarter)
            if label is None:
                continue

            file_path = os.path.join(full_dir_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            chunks = chunk_text(text, max_words=CHUNK_WORDS)

            for chunk_idx, chunk in enumerate(chunks):
                # Minimum 100 words — filters out tiny trailing chunks
                if len(chunk.split()) < 100:
                    continue
                ling = extract_linguistic_features(chunk)
                records.append({
                    "Company":  company,
                    "Year":     year,
                    "Quarter":  quarter,
                    "Chunk_ID": chunk_idx,
                    "Text":     chunk,
                    "Label":    label,
                    **ling
                })

    if not records:
        print("No records found. Check your DATA_DIR and transcript filenames.")
        return

    df = pd.DataFrame(records)

    # Save the full dataset
    full_path = os.path.join(DATASETS_DIR, "longformer_dataset_chunked.csv")
    df.to_csv(full_path, index=False)
    print(f"Full dataset: {len(df)} chunks  |  Saved to {full_path}")

    print("\n--- Class Distribution ---")
    print(df['Label'].value_counts())
    print(df['Label'].value_counts(normalize=True).round(3))

    # Document-level stratified split (prevents leakage)
    df['Doc_ID'] = df['Company'] + '_' + df['Year'].astype(str) + '_Q' + df['Quarter'].astype(str)
    docs_df = df[['Doc_ID', 'Label']].drop_duplicates()

    train_docs, temp_docs = train_test_split(
        docs_df, test_size=0.2, stratify=docs_df['Label'], random_state=42
    )
    val_docs, test_docs = train_test_split(
        temp_docs, test_size=0.5, stratify=temp_docs['Label'], random_state=42
    )

    train_df = df[df['Doc_ID'].isin(train_docs['Doc_ID'])].drop(columns=['Doc_ID'])
    val_df   = df[df['Doc_ID'].isin(val_docs['Doc_ID'])].drop(columns=['Doc_ID'])
    test_df  = df[df['Doc_ID'].isin(test_docs['Doc_ID'])].drop(columns=['Doc_ID'])

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train_longformer.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val_longformer.csv"),   index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test_longformer.csv"), index=False)

    print(f"\nStratified splits (document-level):")
    print(f"  Train : {len(train_df)} chunks from {len(train_docs)} transcripts")
    print(f"  Val   : {len(val_df)} chunks from {len(val_docs)} transcripts")
    print(f"  Test  : {len(test_df)} chunks from {len(test_docs)} transcripts")

    print("\nLabel distribution per split:")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split['Label'].value_counts().to_dict()
        print(f"  {name}: Clean={dist.get(0,0)}  Fraud={dist.get(1,0)}")

if __name__ == "__main__":
    main()
