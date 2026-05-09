"""
generate_report.py
Generates a multi-page PDF project report using matplotlib (no external PDF library needed).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.image as mpimg
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

DATA_DIR      = "/Users/shubhampatthe/Downloads/Data"
REPORTS_DIR   = os.path.join(DATA_DIR, "reports")
CONFUSION_PNG = os.path.join(REPORTS_DIR, "finbertconfusionmatrix.png")
FEATURE_PNG   = os.path.join(REPORTS_DIR, "feature_importance.png")
OUTPUT_PDF    = os.path.join(REPORTS_DIR, "Fraud_Detection_Report.pdf")

DARK_BLUE  = "#1a3c78"
MID_BLUE   = "#2e5db3"
LIGHT_BLUE = "#dce8ff"
ACCENT     = "#e05a00"
BG_WHITE   = "#ffffff"
GRAY_TEXT  = "#444444"

def add_header(fig, title="Corporate Fraud Detection - B.Tech Project Report"):
    fig.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=9, color="#888888", style="italic")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.955, 0.955],
                               transform=fig.transFigure,
                               color="#cccccc", linewidth=0.8))

def add_footer(fig, page_num):
    fig.text(0.5, 0.025, f"Page {page_num}  |  Generated {datetime.now().strftime('%d %B %Y')}",
             ha="center", va="bottom", fontsize=8, color="#aaaaaa")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.04, 0.04],
                               transform=fig.transFigure,
                               color="#cccccc", linewidth=0.8))

def metric_card(ax, x, y, w, h, label, value, bg_color, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.01",
                                facecolor=bg_color, edgecolor="none",
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + w/2, y + h*0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=18, fontweight="bold",
            color=text_color, clip_on=False)
    ax.text(x + w/2, y + h*0.22, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, color=text_color, clip_on=False)

with PdfPages(OUTPUT_PDF) as pdf:

    # ============================================================
    # PAGE 1: TITLE PAGE
    # ============================================================
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG_WHITE)
    add_footer(fig, 1)

    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Title banner
    ax.add_patch(FancyBboxPatch((0.0, 0.74), 1.0, 0.26,
                                boxstyle="square,pad=0",
                                facecolor=DARK_BLUE, edgecolor="none"))
    ax.text(0.5, 0.91, "Corporate Fraud Detection",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.82, "Using a Hybrid NLP Model",
            ha="center", va="center", fontsize=20, color=LIGHT_BLUE)
    ax.text(0.5, 0.76, "FinBERT + Linguistic Feature Engineering",
            ha="center", va="center", fontsize=13, color="#adc8ff")

    # Subtitle block
    ax.text(0.5, 0.69, "B.Tech Final Year Project",
            ha="center", fontsize=12, color=GRAY_TEXT, fontweight="bold")
    ax.text(0.5, 0.65, f"Generated: {datetime.now().strftime('%d %B %Y')}",
            ha="center", fontsize=10, color="#888888")

    # Metric cards
    ax.text(0.5, 0.59, "Model Performance at a Glance",
            ha="center", fontsize=11, color=DARK_BLUE, fontweight="bold")
    metrics = [
        (0.07, "Overall Accuracy", "88%",  "#2e7d32"),
        (0.30, "Fraud Recall",     "96%",  ACCENT),
        (0.53, "Fraud Precision",  "73%",  MID_BLUE),
        (0.76, "Macro F1-Score",   "87%",  "#6a1aaa"),
    ]
    for x, lbl, val, clr in metrics:
        metric_card(ax, x, 0.46, 0.20, 0.12, lbl, val, clr)

    # Summary bullet points
    bullets = [
        "12 companies analysed (6 fraud + 6 non-fraud control)",
        "7,595 text chunks from earnings call transcripts",
        "Document-level stratification prevents data leakage",
        "96% Recall on Fraud class - only 9 fraud chunks missed",
        "Early detection: flagged GE Q1 2015 at 97.78% before SEC action",
    ]
    ax.text(0.08, 0.42, "Key Highlights", fontsize=11,
            color=DARK_BLUE, fontweight="bold")
    for i, b in enumerate(bullets):
        ax.text(0.09, 0.37 - i*0.052, f"  + {b}", fontsize=9.5, color=GRAY_TEXT)

    # Bottom bar
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 0.06,
                                boxstyle="square,pad=0",
                                facecolor=DARK_BLUE, edgecolor="none"))
    ax.text(0.5, 0.03, "Hybrid FinBERT Fraud Detector  |  NLP Research Project",
            ha="center", va="center", fontsize=9, color=LIGHT_BLUE)

    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ============================================================
    # PAGE 2: INTRODUCTION & DATASET
    # ============================================================
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG_WHITE)
    add_header(fig); add_footer(fig, 2)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.87], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.97
    def heading(text, ypos, size=13):
        ax.add_patch(FancyBboxPatch((0, ypos-0.025), 1.0, 0.038,
                                    boxstyle="round,pad=0.005",
                                    facecolor=LIGHT_BLUE, edgecolor="none"))
        ax.text(0.01, ypos, text, fontsize=size, fontweight="bold",
                color=DARK_BLUE, va="center")
        return ypos - 0.055

    def para(text, ypos, size=9.2):
        wrapped = []
        line = ""
        for word in text.split():
            test = line + " " + word if line else word
            if len(test) > 100:
                wrapped.append(line)
                line = word
            else:
                line = test
        if line:
            wrapped.append(line)
        for w in wrapped:
            ax.text(0.01, ypos, w, fontsize=size, color=GRAY_TEXT, va="top")
            ypos -= 0.038
        return ypos - 0.01

    y = heading("1. Introduction & Motivation", y)
    intro = ("Corporate accounting fraud causes enormous financial and social damage. Traditional "
             "detection relies on quantitative financial-ratio analysis which only flags anomalies "
             "after fraudulent transactions are recorded. This project proposes an NLP-based "
             "early-detection system that analyses the language of corporate executives during "
             "quarterly earnings calls to identify deceptive behaviour before fraud is uncovered.")
    y = para(intro, y)

    contrib = ("We fine-tune ProsusAI/FinBERT and augment its contextual embeddings with six "
               "hand-crafted linguistic features known to correlate with deceptive speech, "
               "achieving 96% recall on the Fraud class.")
    y = para(contrib, y)

    y -= 0.01
    y = heading("2. Dataset & Labeling Strategy", y)

    companies_text = ("Transcripts for 12 companies were collected over multiple years. "
                      "6 confirmed fraud companies were matched with 6 non-fraud control companies "
                      "from the same industry sector.")
    y = para(companies_text, y)

    # Company table
    y -= 0.01
    ax.text(0.01, y, "Company Selection", fontsize=10, fontweight="bold", color=DARK_BLUE); y -= 0.035
    cols = [("Fraud Companies", 0.01, 0.48), ("Non-Fraud Control", 0.50, 0.48)]
    ax.add_patch(FancyBboxPatch((0, y-0.005), 1.0, 0.028,
                                boxstyle="square,pad=0", facecolor=DARK_BLUE, edgecolor="none"))
    for lbl, x, w in cols:
        ax.text(x + 0.01, y + 0.007, lbl, fontsize=9, fontweight="bold",
                color="white", va="center")
    y -= 0.032

    pairs = [
        ("General Electric",          "3M"),
        ("MiMedx",                    "Integra LifeSciences"),
        ("Satyam Computer Services",  "Tata Consultancy Services"),
        ("Under Armour",              "Johnson & Johnson"),
        ("Valeant Pharmaceuticals",   "Nike"),
        ("Weatherford International", "Schlumberger"),
    ]
    for i, (f, c) in enumerate(pairs):
        bg = LIGHT_BLUE if i % 2 == 0 else "white"
        ax.add_patch(FancyBboxPatch((0, y-0.005), 1.0, 0.026,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor="none"))
        ax.text(0.02, y + 0.006, f, fontsize=8.5, color=GRAY_TEXT)
        ax.text(0.52, y + 0.006, c, fontsize=8.5, color=GRAY_TEXT)
        y -= 0.030

    y -= 0.01
    y = heading("3-Zone Labeling Strategy", y, size=11)
    zones = [
        ("Zone A (label = 1)",         "Transcripts during the confirmed SEC fraud window  -->  used in training."),
        ("Zone B (label = EXCLUDED)",  "Pre-fraud transcripts from fraud companies  -->  used for early-detection only."),
        ("Zone C (label = 0)",         "All transcripts from non-fraud control companies  -->  used in training."),
    ]
    for zone, desc in zones:
        ax.text(0.01, y, zone + ":", fontsize=9, fontweight="bold", color=ACCENT); 
        ax.text(0.32, y, desc, fontsize=9, color=GRAY_TEXT)
        y -= 0.040

    y -= 0.005
    chunk_txt = ("Each transcript was split into 350-word chunks (to fit FinBERT's 512-token limit), "
                 "yielding 7,595 chunks: 6,037 train / 847 validation / 711 test, with "
                 "document-level stratification to prevent data leakage.")
    y = para(chunk_txt, y)

    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ============================================================
    # PAGE 3: METHODOLOGY
    # ============================================================
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG_WHITE)
    add_header(fig); add_footer(fig, 3)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.87], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.97
    y = heading("4. Methodology", y)

    stages = [
        ("Stage 1 - Data Ingestion & Labeling",
         "Raw transcript text files were parsed, date-stamped, and assigned binary labels "
         "based on 3-Zone boundaries for each company."),
        ("Stage 2 - Linguistic Feature Engineering",
         "Six explicit deception markers computed per chunk: Hedging Ratio (might, approximately...), "
         "Complexity Ratio (legalistic jargon), Average Sentence Length, Unique Word Ratio, "
         "Question Count, and Number Ratio."),
        ("Stage 3 - Hybrid Model Architecture",
         "HybridFraudDetector: ProsusAI/FinBERT encoder produces a 768-dim [CLS] embedding. "
         "A 64-dim projection of the 6 linguistic features is concatenated, giving an 832-dim "
         "vector. A 2-layer classifier (832->256->2) with 0.5 dropout produces the final logits. "
         "Weighted Cross-Entropy loss (sklearn compute_class_weight) corrects for class imbalance."),
        ("Stage 4 - Training & Evaluation",
         "5 epochs, AdamW lr=2e-5, fp16 mixed precision, Kaggle T4x2 GPUs. "
         "Evaluation: Precision, Recall, F1 and Confusion Matrix on the hold-out test set."),
    ]
    for stg, desc in stages:
        ax.add_patch(FancyBboxPatch((0, y-0.028), 1.0, 0.036,
                                    boxstyle="round,pad=0.005",
                                    facecolor="#fff3e0", edgecolor=ACCENT, linewidth=0.5))
        ax.text(0.01, y, stg, fontsize=9.5, fontweight="bold", color=ACCENT, va="center")
        y -= 0.055
        y = para(desc, y)
        y -= 0.005

    # Architecture diagram text
    y -= 0.01
    y = heading("Model Architecture Diagram", y, size=11)
    arch_lines = [
        "Input: Earnings Call Transcript Chunk (up to 512 tokens)",
        "       |",
        "+------+--------------------------------------+",
        "| FinBERT Encoder  (12 transformer layers)   |   6 Linguistic Features",
        "|   Local + Global self-attention             |   (hedge, complexity,  ",
        "|   [CLS] token representation (768-dim)      |    sentence_len, etc.) ",
        "+------+--------------------------------------+   +-------------------+",
        "       |                                              |",
        "       +------------------+---------------------------+",
        "                          | Concatenate (832-dim)     |",
        "                     +----+------+",
        "                     | FC 832->256 | ReLU | Dropout(0.5)|",
        "                     +----+------+",
        "                          |",
        "                     +----+------+",
        "                     | FC 256->2  | Softmax            |",
        "                     +----+------+",
        "                          |",
        "              P(Clean)  P(Fraud)  -->  Verdict",
    ]
    ax.add_patch(FancyBboxPatch((0, y - len(arch_lines)*0.028 - 0.01), 1.0,
                                len(arch_lines)*0.028 + 0.02,
                                boxstyle="round,pad=0.01",
                                facecolor="#f0f4ff", edgecolor=MID_BLUE, linewidth=0.7))
    for line in arch_lines:
        ax.text(0.03, y, line, fontsize=7.5, color=DARK_BLUE,
                fontfamily="monospace", va="top")
        y -= 0.028

    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ============================================================
    # PAGE 4: RESULTS (Confusion Matrix)
    # ============================================================
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG_WHITE)
    add_header(fig); add_footer(fig, 4)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.87], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.97
    y = heading("5. Results", y)

    # Classification report table
    ax.text(0.01, y, "Classification Report - Unseen Test Set (711 chunks)",
            fontsize=10, fontweight="bold", color=DARK_BLUE); y -= 0.038

    cols_h = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    xs     = [0.0, 0.20, 0.40, 0.60, 0.80]
    ws     = [0.20, 0.20, 0.20, 0.20, 0.20]
    ax.add_patch(FancyBboxPatch((0, y-0.008), 1.0, 0.032,
                                boxstyle="square,pad=0", facecolor=DARK_BLUE, edgecolor="none"))
    for h, x in zip(cols_h, xs):
        ax.text(x + 0.01, y + 0.006, h, fontsize=9, fontweight="bold",
                color="white", va="center")
    y -= 0.038

    table_data = [
        ("Clean (0)",    "0.98", "0.85", "0.91", "505", "#e8f5e9"),
        ("Fraud (1)",    "0.73", "0.96", "0.83", "206", "#fff3e0"),
        ("Accuracy",     "",     "",     "0.88",  "711", LIGHT_BLUE),
        ("Macro Avg",    "0.85", "0.90", "0.87", "711", "#f5f5f5"),
        ("Weighted Avg", "0.91", "0.88", "0.89", "711", "#f5f5f5"),
    ]
    for row in table_data:
        vals, bg = row[:-1], row[-1]
        ax.add_patch(FancyBboxPatch((0, y-0.008), 1.0, 0.030,
                                    boxstyle="square,pad=0", facecolor=bg, edgecolor="none"))
        for v, x in zip(vals, xs):
            ax.text(x + 0.01, y + 0.005, v, fontsize=9, color=GRAY_TEXT, va="center")
        y -= 0.035

    y -= 0.01
    result_text = ("The model achieves 96% recall on the Fraud class, successfully identifying "
                   "197 out of 206 fraudulent chunks. Only 9 fraud chunks were missed (False Negatives). "
                   "The overall accuracy is 88% on the unseen hold-out test set.")
    y = para(result_text, y)

    # Embed confusion matrix image
    ax.text(0.5, y, "Confusion Matrix", fontsize=10, fontweight="bold",
            color=DARK_BLUE, ha="center"); y -= 0.015
    if os.path.exists(CONFUSION_PNG):
        img_ax = fig.add_axes([0.20, y - 0.38, 0.60, 0.38], frameon=False)
        img = mpimg.imread(CONFUSION_PNG)
        img_ax.imshow(img); img_ax.axis("off")
        y -= 0.40
    else:
        ax.text(0.5, y, "[confusion_matrix.png not found]",
                ha="center", fontsize=9, color="red"); y -= 0.05

    pdf.savefig(fig, bbox_inches="tight"); plt.close()

    # ============================================================
    # PAGE 5: FEATURE IMPORTANCE + EARLY DETECTION
    # ============================================================
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG_WHITE)
    add_header(fig); add_footer(fig, 5)
    ax = fig.add_axes([0.07, 0.06, 0.86, 0.87], frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.97
    y = heading("6. Model Interpretability - Linguistic Feature Importance", y)

    interp_text = ("To interpret the model's decisions, we analysed the absolute weight magnitudes "
                   "of the first layer of the linguistic projection network (64x6). Summing across "
                   "the 64 output neurons gives a relative importance score for each of the 6 features.")
    y = para(interp_text, y)

    if os.path.exists(FEATURE_PNG):
        img_ax2 = fig.add_axes([0.05, y - 0.31, 0.90, 0.30], frameon=False)
        img2 = mpimg.imread(FEATURE_PNG)
        img_ax2.imshow(img2); img_ax2.axis("off")
        y -= 0.33

    finding = ("Key finding: Average Sentence Length (18.8%) is the strongest deception signal. "
               "Fraudulent executives use long, rambling sentences to dilute negative information. "
               "Hedging Ratio (16.5%) and Unique Word Ratio (18.0%) follow closely.")
    y = para(finding, y)

    y -= 0.01
    y = heading("7. Early-Detection Experiment (Zone B Validation)", y)

    early_text = ("To demonstrate prospective utility, we applied the inference pipeline to a "
                  "pre-fraud transcript from General Electric (Q1 2015) - a period BEFORE the "
                  "SEC officially confirmed fraudulent activity.")
    y = para(early_text, y)

    # Early detection result box
    ax.add_patch(FancyBboxPatch((0.05, y-0.11), 0.90, 0.12,
                                boxstyle="round,pad=0.01",
                                facecolor="#fff3e0", edgecolor=ACCENT, linewidth=1.2))
    ax.text(0.50, y - 0.01, "! HIGH RISK DETECTED", ha="center", fontsize=11,
            fontweight="bold", color=ACCENT)
    ax.text(0.50, y - 0.04, "General Electric - Q1 2015 (Pre-Fraud Period)", ha="center",
            fontsize=9.5, color=GRAY_TEXT)
    ax.text(0.50, y - 0.065, "Overall Fraud Probability: 97.78%",
            ha="center", fontsize=12, fontweight="bold", color=DARK_BLUE)
    ax.text(0.50, y - 0.09, "Highest Risk Chunk: 99.37%",
            ha="center", fontsize=9, color=GRAY_TEXT)
    y -= 0.14

    conclusion_text = ("The model flagged this pre-fraud transcript with 97.78% fraud probability - "
                       "more than a year before the SEC enforcement action. This demonstrates the "
                       "model's ability to detect deceptive linguistic patterns prospectively, "
                       "offering a novel early-warning tool for auditors and regulators.")
    y = para(conclusion_text, y)

    y -= 0.01
    y = heading("8. Conclusion & Future Work", y)

    conc = ("This project demonstrates that a Hybrid NLP model combining FinBERT contextual "
            "embeddings with explicit linguistic feature engineering can detect corporate fraud "
            "with high accuracy (88%) and exceptional fraud recall (96%). The early-detection "
            "experiment shows the model can flag deceptive language before fraud is officially "
            "uncovered, offering a novel tool for auditors, investors, and regulators.")
    y = para(conc, y)

    future = [
        "1. Expand to 30+ companies for greater generalisability.",
        "2. Replace FinBERT with Longformer to process full transcripts without chunking.",
        "3. Deploy as a real-time API for auditors to query with any earnings-call PDF.",
        "4. Integrate financial ratio features alongside NLP for a multi-modal model.",
    ]
    for f in future:
        ax.text(0.03, y, f, fontsize=9, color=GRAY_TEXT); y -= 0.040

    pdf.savefig(fig, bbox_inches="tight"); plt.close()

print(f"Report saved to: {OUTPUT_PDF}")
