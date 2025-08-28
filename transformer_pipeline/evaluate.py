import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model, X_val, y_val, label_cols, report_dir="Reports/Visuals"):
    """
    Evaluate BERT model and generate classification report + confusion matrix.
    
    Args:
        model: Trained keras model
        X_val: dict with {"input_ids": ..., "attention_mask": ...}
        y_val: true labels (numpy array or tensor)
        label_cols: list of label names
        report_dir: folder to save confusion matrix plot
    """
    # Ensure output directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Predictions
    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int)

    # Classification report
    print("\n📊 Classification Report:\n")
    print(classification_report(y_val, preds, target_names=label_cols))

    # Confusion matrix for "toxic" label only (index 0)
    cm = confusion_matrix(y_val[:, 0], preds[:, 0])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Toxic")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    save_path = os.path.join(report_dir, "Confusion_Matrix_Toxic.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"✅ Confusion matrix saved at: {save_path}")
