import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Feature names matching the order in the model
FEATURES = [
    'Hedging Ratio', 
    'Complexity Ratio', 
    'Avg Sentence Length', 
    'Unique Word Ratio', 
    'Question Count', 
    'Number Ratio'
]

# ---------------------------------------------------------
# LOAD THE MODEL
# ---------------------------------------------------------
class HybridFraudDetector(nn.Module):
    def __init__(self, n_linguistic_features=6):
        super().__init__()
        # We only need the linguistic projection and classifier layers for this analysis
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

def analyze_feature_importance():
    model_path = "/Users/shubhampatthe/Downloads/Data/models/best_hybrid_model.pt"
    if not os.path.exists(model_path):
        print("Model file not found.")
        return

    model = HybridFraudDetector()
    state_dict = torch.load(model_path, map_location='cpu')
    
    # Filter state dict for only the layers we need (ignoring BERT to save memory/time)
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        if "linguistic_proj" in k or "classifier" in k:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    
    print("Extracting weights from the Linguistic Neural Network...")
    
    # ---------------------------------------------------------
    # WEIGHT MAGNITUDE ANALYSIS
    # ---------------------------------------------------------
    # The first layer connects the 6 features to 64 hidden neurons.
    # The shape is [64, 6] (Out_features, In_features)
    # We take the absolute value of all weights, and sum them along the 64 neurons
    # to see which of the 6 input features has the strongest connection to the network.
    first_layer_weights = model.linguistic_proj[0].weight.data.numpy()
    
    # Sum of absolute weights for each feature
    importance_scores = np.sum(np.abs(first_layer_weights), axis=0)
    
    # Normalize to percentages
    importance_scores = (importance_scores / np.sum(importance_scores)) * 100
    
    # Sort features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [FEATURES[i] for i in sorted_indices]
    sorted_scores = [importance_scores[i] for i in sorted_indices]
    
    print("\n--- Linguistic Feature Importance (Weight Analysis) ---")
    for feat, score in zip(sorted_features, sorted_scores):
        print(f"{feat:25s} {score:5.2f}%")
        
    # ---------------------------------------------------------
    # PLOT RESULTS
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(FEATURES)))
    bars = plt.barh(sorted_features[::-1], sorted_scores[::-1], color=colors)
    plt.xlabel('Relative Importance (%)')
    plt.title('Hybrid Model: Linguistic Feature Importance in Fraud Detection')
    
    # Add percentage labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                 f'{width:.1f}%', ha='left', va='center')
                 
    plt.tight_layout()
    output_png = "/Users/shubhampatthe/Downloads/Data/reports/feature_importance.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n✅ Visualization saved to {output_png}")

if __name__ == "__main__":
    analyze_feature_importance()
