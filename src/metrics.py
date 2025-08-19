from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score
)

def evaluate(y_true, y_pred, y_proba: Optional[np.ndarray] = None) -> Tuple[float, float, float, float, Optional[float]]:
    """Return (acc, prec, rec, f1, roc_auc or None) and print a short report."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    roc = None
    if y_proba is not None:
        roc = roc_auc_score(y_true, y_proba)
        print(f"ROC-AUC  : {roc:.4f}")
    return acc, prec, rec, f1, roc
