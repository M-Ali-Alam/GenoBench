from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc

class ClassificationEvaluator:
    """Accumulates predictions and computes biological classification benchmarks."""
    
    def __init__(self) -> None:
        self.true_labels: List[int] = []
        self.predictions: List[float] = []

    def update(self, true_label: int, prediction: float) -> None:
        """Append a single sample execution outcome to the state."""
        self.true_labels.append(true_label)
        self.predictions.append(prediction)

    def compute_metrics(self) -> Dict[str, float]:
        """Calculates final metrics across all accumulated samples."""
        if not self.true_labels:
            raise ValueError("No metrics to compute. Accumulator state is empty.")

        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Convert continuous probabilities to binary decisions at 0.5 threshold for MCC
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        
        # Calculate AUPRC (Area Under Precision-Recall Curve)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall, precision)

        return {
            "mcc": float(mcc),
            "auprc": float(auprc),
            "sample_count": len(self.true_labels)
        }