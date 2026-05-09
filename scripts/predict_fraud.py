import os
import re
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import warnings

# Suppress HuggingFace warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# LINGUISTIC FEATURES SETUP
# ---------------------------------------------------------
HEDGE_WORDS = ['approximately', 'perhaps', 'might', 'could', 'possibly', 
               'uncertain', 'believe', 'expect', 'roughly', 'around', 
               'may', 'seems', 'appears', 'suggest', 'likely']

COMPLEXITY_WORDS = ['notwithstanding', 'pursuant', 'accordingly', 'thereafter', 
                    'hereinafter', 'whereby', 'aforementioned']

def extract_features_for_chunk(text):
    text_str = str(text).lower()
    words = text_str.split()
    total = len(words) + 1e-9
    sentences = [s for s in re.split(r'[.!?]', text_str) if len(s.strip()) > 0]
    
    return [
        sum(w in HEDGE_WORDS for w in words) / total,
        sum(w in COMPLEXITY_WORDS for w in words) / total,
        total / (len(sentences) + 1e-9),
        len(set(words)) / total,
        text_str.count('?'),
        sum(bool(re.search(r'\d', w)) for w in words) / total,
    ]

# ---------------------------------------------------------
# MODEL ARCHITECTURE (Must match exactly with training script)
# ---------------------------------------------------------
class HybridFraudDetector(nn.Module):
    def __init__(self, n_linguistic_features=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # We don't strictly need class weights for inference, but we keep the buffer 
        # so loading the state_dict doesn't throw a "Missing key" error
        self.register_buffer('class_weights_tensor', torch.tensor([1.0, 1.0], dtype=torch.float32))
        
        self.linguistic_proj = nn.Sequential(
            nn.Linear(n_linguistic_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
    def forward(self, input_ids, attention_mask, linguistic_features, token_type_ids=None):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_embed = bert_out.last_hidden_state[:, 0, :]
        ling_embed = self.linguistic_proj(linguistic_features.float())
        combined = torch.cat((cls_embed, ling_embed), dim=1)
        logits = self.classifier(combined)
        return logits

# ---------------------------------------------------------
# INFERENCE PIPELINE
# ---------------------------------------------------------
def predict_transcript(file_path, model_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights '{model_path}' not found.")
        print("Please download 'best_hybrid_model.pt' from Kaggle and place it here.")
        return

    print(f"\n📄 Analyzing Transcript: {os.path.basename(file_path)}")
    print("⏳ Loading Hybrid Fraud Detector...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridFraudDetector()
    
    # Load weights safely, handling potential DataParallel 'module.' prefixes from Kaggle
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # Remove 'module.' prefix if saved with DataParallel
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    
    print("✂️  Chunking text and extracting linguistic features...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        full_text = f.read()
        
    # Chunking logic (max 350 words)
    words = full_text.split()
    chunks = [" ".join(words[i:i+350]) for i in range(0, len(words), 350)]
    
    # Filter out empty or tiny chunks
    chunks = [c for c in chunks if len(c.split()) > 20]
    
    if not chunks:
        print("❌ Error: Transcript is too short or empty.")
        return

    chunk_scores = []
    
    print(f"🧠 Running {len(chunks)} chunks through the Hybrid Model...")
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            # 1. Linguistic Features
            ling_feats = extract_features_for_chunk(chunk)
            ling_tensor = torch.tensor([ling_feats]).to(device)
            
            # 2. Tokenization
            inputs = tokenizer(
                chunk, 
                padding="max_length", 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(device)
            
            # 3. Model Prediction
            logits = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids", None),
                linguistic_features=ling_tensor
            )
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)[0]
            fraud_prob = probs[1].item() # Probability of Class 1 (Fraud)
            chunk_scores.append((fraud_prob, chunk))
            
    # Aggregate results
    avg_fraud_prob = np.mean([score for score, _ in chunk_scores])
    max_fraud_prob = np.max([score for score, _ in chunk_scores])
    
    print("\n" + "="*50)
    print("📊 FRAUD RISK ANALYSIS REPORT")
    print("="*50)
    print(f"Overall Document Fraud Probability: {avg_fraud_prob*100:.2f}%")
    print(f"Highest Risk Chunk Probability:     {max_fraud_prob*100:.2f}%")
    
    # Classification Threshold (can be adjusted)
    if avg_fraud_prob > 0.40:
        print("\n⚠️  FINAL VERDICT: HIGH RISK OF DECEPTION DETECTED ⚠️")
    else:
        print("\n✅ FINAL VERDICT: CLEAN / LOW RISK")

    print("\n--- Most Suspicious Quote from the Call ---")
    # Sort chunks by fraud probability descending
    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    top_score, top_chunk = chunk_scores[0]
    
    # Print an excerpt of the most suspicious chunk
    excerpt = top_chunk[:500] + "..." if len(top_chunk) > 500 else top_chunk
    print(f"(Risk Score: {top_score*100:.2f}%)\n\"{excerpt}\"")
    print("==================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze an earnings call transcript for fraud.')
    parser.add_argument('--file', type=str, required=True, help='Path to the .txt transcript file')
    parser.add_argument('--model', type=str, default='/Users/shubhampatthe/Downloads/Data/models/best_hybrid_model.pt', help='Path to the trained PyTorch model (.pt file)')
    args = parser.add_argument_group()
    
    # Simple hardcoded check if no args provided (makes it easier to run in IDEs)
    if len(sys.argv) == 1:
        print("Usage: python predict_fraud.py --file <path_to_transcript.txt>")
        print("Example: python predict_fraud.py --file '../transcripts/3M/Q_1_ 2021.txt'")
    else:
        args = parser.parse_args()
        predict_transcript(args.file, args.model)
