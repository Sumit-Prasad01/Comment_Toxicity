import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_val, y_val, label_cols):

    preds = model.predict(X_val)
    preds = (preds > 0.5).astype(int)

    print("\n Classification Report: \n")
    print(classification_report(y_val, preds, target_names = label_cols))

    # Confusion matrix for "Toxic" label 
    cm = confusion_matrix(y_val[:,0], preds[:,0])
    sns.heatmap(cm, annot = True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Toxic")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("../Reports/Visuals/Confusion Matrix - Toxic.png")
    plt.show()


