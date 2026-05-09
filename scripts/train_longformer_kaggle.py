import os
import torch
import torch.nn as nn

# ---------------------------------------------------------
# FIX: Longformer crashes under PyTorch DataParallel due to a 
# generator StopIteration in its parameter iteration.
# Since Kaggle automatically uses DataParallel for T4 x2, we 
# completely bypass it by mocking nn.DataParallel to act as a 
# transparent wrapper. This forces single-GPU training safely.
# ---------------------------------------------------------
class DummyDataParallel(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

nn.DataParallel = DummyDataParallel
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# SETUP — UPDATE INPUT_DIR AFTER UPLOADING TO KAGGLE
# For testing locally, you can change it to: /Users/shubhampatthe/Downloads/Data/datasets/longformer
INPUT_DIR  = "/kaggle/input/datasets/shubhamp06/longformer-dataset/"
OUTPUT_DIR = "/kaggle/working/longformer-fraud-model"

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_longformer.csv"))
val_df   = pd.read_csv(os.path.join(INPUT_DIR, "val_longformer.csv"))
test_df  = pd.read_csv(os.path.join(INPUT_DIR, "test_longformer.csv"))
print(f"Datasets loaded  —  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------------------------------------------
# LINGUISTIC FEATURES (identical to FinBERT model for fair comparison)
# ---------------------------------------------------------
LINGUISTIC_FEATURES = [
    'hedge_ratio', 'complexity_ratio', 'avg_sentence_len',
    'unique_word_ratio', 'question_count', 'number_ratio'
]

# ---------------------------------------------------------
# MODEL & TOKENIZER
# ---------------------------------------------------------
MODEL_NAME = "allenai/longformer-base-4096"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN    = 1792   # safe upper limit for T4 x2 with batch_size=4

# ---------------------------------------------------------
# TOKENISATION — with Longformer global attention on [CLS]
# ---------------------------------------------------------
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["Text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

    # Longformer REQUIRES a global_attention_mask.
    # We set 1 on the [CLS] token (index 0) and 0 everywhere else.
    batch_size = len(examples["Text"])
    global_attn = [[0] * MAX_LEN for _ in range(batch_size)]
    for i in range(batch_size):
        global_attn[i][0] = 1   # [CLS] gets global attention
    tokenized["global_attention_mask"] = global_attn

    # Pack linguistic features
    ling_feats = []
    for i in range(batch_size):
        feats = [examples[feat][i] for feat in LINGUISTIC_FEATURES]
        ling_feats.append(feats)
    tokenized["linguistic_features"] = ling_feats

    return tokenized

print("Tokenising datasets...")
train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)
test_dataset  = Dataset.from_pandas(test_df)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val   = val_dataset.map(tokenize_function,   batched=True)
tokenized_test  = test_dataset.map(tokenize_function,  batched=True)

# Rename label column
for ds in [tokenized_train, tokenized_val, tokenized_test]:
    pass
tokenized_train = tokenized_train.rename_column("Label", "labels")
tokenized_val   = tokenized_val.rename_column("Label", "labels")
tokenized_test  = tokenized_test.rename_column("Label", "labels")

# Set torch format
COLS = ["input_ids", "attention_mask", "global_attention_mask", "labels", "linguistic_features"]
tokenized_train.set_format("torch", columns=COLS)
tokenized_val.set_format("torch",   columns=COLS)
tokenized_test.set_format("torch",  columns=COLS)

# ---------------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------------
labels_array  = train_df["Label"].values
class_weights = compute_class_weight("balanced",
                                     classes=np.unique(labels_array),
                                     y=labels_array)
print(f"Class weights: {class_weights}")

# ---------------------------------------------------------
# HYBRID LONGFORMER ARCHITECTURE
# ---------------------------------------------------------
class HybridLongformerDetector(nn.Module):
    def __init__(self, n_linguistic_features=6):
        super().__init__()
        self.longformer = AutoModel.from_pretrained(MODEL_NAME)

        # Class weights stored as a buffer so DataParallel moves them automatically
        self.register_buffer(
            "class_weights_tensor",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        # Linguistic sub-network
        self.linguistic_proj = nn.Sequential(
            nn.Linear(n_linguistic_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(768 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        self.supports_gradient_checkpointing = True

    def gradient_checkpointing_enable(self, **kwargs):
        self.longformer.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.longformer.gradient_checkpointing_disable(**kwargs)

    def forward(self, input_ids, attention_mask, global_attention_mask,
                linguistic_features, labels=None):
        # Longformer forward pass — must pass global_attention_mask
        out = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_embed  = out.last_hidden_state[:, 0, :]          # [CLS] token
        ling_embed = self.linguistic_proj(linguistic_features.float())
        combined   = torch.cat((cls_embed, ling_embed), dim=1)
        logits     = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
            loss    = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits} if labels is not None else {"logits": logits}

model = HybridLongformerDetector(n_linguistic_features=len(LINGUISTIC_FEATURES))

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# ---------------------------------------------------------
# TRAINING — smaller batch size because Longformer uses more GPU memory
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,      # Reduced to 1 to completely prevent OOM on T4
    gradient_accumulation_steps=4,      # Accumulate 4 steps (effective batch size = 4)
    per_device_eval_batch_size=2,       # Reduced eval batch size
    gradient_checkpointing=True,        # Massively reduces VRAM usage by trading compute for memory
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=False,   # must be False when bypassing DataParallel on T4 x2
    logging_dir="/kaggle/working/logs",
    logging_steps=50,
    report_to="none",
)

# SingleGPUTrainer bypasses DataParallel — required for Longformer on Kaggle T4 x2
# We override both _wrap_model AND get_model_output for belt-and-suspenders safety.
class SingleGPUTrainer(Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        # Return the raw model — skip DataParallel / DistributedDataParallel wrapping
        return model

    def _move_model_to_device(self, model, device):
        # Move model to single device (cuda:0) explicitly
        return model.to(device)

trainer = SingleGPUTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Starting Longformer Hybrid Model Training...")
    trainer.train()

    print("\n--- Evaluating on Test Set ---")
    predictions = trainer.predict(tokenized_test)
    preds  = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("\n--- Detailed Classification Report ---")
    print(classification_report(labels, preds, target_names=["Clean (0)", "Fraud (1)"]))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean", "Fraud"],
                yticklabels=["Clean", "Fraud"])
    plt.title("Confusion Matrix - Hybrid Longformer Fraud Detector")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.savefig("/kaggle/working/longformer_confusion_matrix.png")
    plt.show()

    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "best_longformer_model.pt"))
    print("Done! Download best_longformer_model.pt and longformer_confusion_matrix.png from Output.")
