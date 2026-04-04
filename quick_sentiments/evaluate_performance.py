import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score

def evaluate_performance(y_true, y_prob, positive_label=1):
    """
    Evaluates model performance across different probability thresholds.
    
    Args:
        y_true (list or np.array): The actual ground truth labels.
        y_prob (list or np.array): The predicted probabilities for the positive class.
        positive_label (int/str): The value representing the positive class (default is 1).
        
    Returns:
        dict: Contains optimal thresholds and a pandas DataFrame of the decile summary.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # 1. Best ROC Threshold (Youden's J statistic: TPR - FPR)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob, pos_label=positive_label)
    youden_j = tpr - fpr
    best_roc_idx = np.argmax(youden_j)
    best_roc_threshold = roc_thresh[best_roc_idx]
    
    # 2. Best PR Threshold (Max F1-Score)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob, pos_label=positive_label)
    # Ignore warnings for 0 division
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores) # Convert NaNs to 0
    best_pr_idx = np.argmax(f1_scores)
    # precision_recall_curve returns thresholds len - 1, so we handle the index safely
    best_pr_threshold = pr_thresh[best_pr_idx] if best_pr_idx < len(pr_thresh) else 1.0

    # 3. Decile Summary Table (0.0 to 1.0)
    summary_data = []
    thresholds = np.arange(0.0, 1.1, 0.1)
    
    for th in thresholds:
        # Convert probabilities to hard predictions based on current threshold
        y_pred = (y_prob >= th).astype(int)
        is_pos = (y_true == positive_label).astype(int)
        
        tp = np.sum((y_pred == 1) & (is_pos == 1))
        fp = np.sum((y_pred == 1) & (is_pos == 0))
        tn = np.sum((y_pred == 0) & (is_pos == 0))
        fn = np.sum((y_pred == 0) & (is_pos == 1))
        
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        summary_data.append({
            "Threshold": round(th, 1),
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3)
        })
        
    df_summary = pd.DataFrame(summary_data)
    
    return {
        "best_roc_threshold": round(best_roc_threshold, 4),
        "best_pr_threshold": round(best_pr_threshold, 4),
        "decile_table": df_summary
    }