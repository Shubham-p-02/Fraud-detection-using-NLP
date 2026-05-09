import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch import nn
from datasets import Dataset

# ---------------------------------------------------------
# SETUP FOR KAGGLE ENVIRONMENT
# ---------------------------------------------------------
# In Kaggle, datasets are typically loaded from /kaggle/input/
# Update this path to match your Kaggle dataset name.
# Example: If your dataset is named 'fraud-earnings-calls', the path is:
# INPUT_DIR = "/kaggle/input/fraud-earnings-calls/"

# For testing locally, you can change this back to your local path. (e.g. /Users/shubhampatthe/Downloads/Data/datasets/finbert)
INPUT_DIR = "/kaggle/input/your-dataset-name-here/"
OUTPUT_DIR = "/kaggle/working/finbert-fraud-model"

# Load datasets
try:
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_dataset.csv"))
    val_df = pd.read_csv(os.path.join(INPUT_DIR, "val_dataset.csv"))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_dataset.csv"))
    print("Datasets loaded successfully!")
except FileNotFoundError:
    print(f"Error: Could not find datasets in {INPUT_DIR}.")
    print("Please make sure you have uploaded the CSVs to Kaggle and updated the INPUT_DIR path.")
    # Exiting early if data isn't found (Kaggle will show this error)
    import sys
    sys.exit(1)

# Convert Pandas DataFrames to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# ---------------------------------------------------------
# MODEL & TOKENIZER INITIALIZATION
# ---------------------------------------------------------
# We use ProsusAI's FinBERT, which is heavily pre-trained on financial text
MODEL_NAME = "ProsusAI/finbert"

print(f"Loading Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# FinBERT originally has 3 labels (positive, negative, neutral). 
# We ignore the original classification head and initialize a new one for binary classification.
print(f"Loading Model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2, 
    ignore_mismatched_sizes=True
)

# ---------------------------------------------------------
# TOKENIZATION FUNCTION
# ---------------------------------------------------------
def tokenize_function(examples):
    # Truncation and Padding are handled here. Max length is 512 for BERT.
    return tokenizer(
        examples["Text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Rename 'Label' to 'labels' as expected by PyTorch/HuggingFace
tokenized_train = tokenized_train.rename_column("Label", "labels")
tokenized_val = tokenized_val.rename_column("Label", "labels")
tokenized_test = tokenized_test.rename_column("Label", "labels")

# Set the format to PyTorch tensors and only keep the necessary columns
columns_to_keep = ['input_ids', 'attention_mask', 'labels']
if 'token_type_ids' in tokenized_train.column_names:
    columns_to_keep.append('token_type_ids')

tokenized_train.set_format("torch", columns=columns_to_keep)
tokenized_val.set_format("torch", columns=columns_to_keep)
tokenized_test.set_format("torch", columns=columns_to_keep)

# ---------------------------------------------------------
# METRICS COMPUTATION
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate precision, recall, and f1 for the minority class (fraud = 1) and overall macro
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ---------------------------------------------------------
# TRAINING SETUP
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16, # Adjust to 8 if you hit Out-Of-Memory (OOM) on Kaggle GPUs
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",      # Optimize for F1 score, crucial for fraud detection
    fp16=True,                       # Enables mixed precision training (faster on modern GPUs like Kaggle's T4)
    logging_dir="/kaggle/working/logs",
    logging_steps=50,
)

# Calculate class weights dynamically
labels_array = train_df['Label'].values
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_array),
    y=labels_array
)
print(f"Calculated Class Weights: {class_weights_array}")

# Custom Trainer to apply class weights
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    class_weights=class_weights_array,
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# EXECUTE TRAINING
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Training...")
    trainer.train()

    print("\n--- Evaluating on Test Set ---")
    results = trainer.evaluate(tokenized_test)
    print(results)

    print("\n--- Generating Detailed Classification Report ---")
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    
    report = classification_report(labels, preds, target_names=["Clean (0)", "Fraud (1)"])
    print(report)

    # Save final model
    print("Saving the best model to Kaggle output directory...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_fraud_model"))
    print("Done! You can now download the best model from the Kaggle Output section.")
