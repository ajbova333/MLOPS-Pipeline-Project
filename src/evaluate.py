"""" Evaluation script for the heart disease prediction model. """

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def compute_metrics(y_true, y_pred, y_proba=None):
    """ Computes evaluation metrics for the model predictions. 
    Returns a dictionary containing accuracy, precision, recall, f1-score, and confusion matrix. """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics

def meets_thresholds(metrics, thresholds):
    """ Checks if the computed metrics meet the specified thresholds. 
    Returns True if all metrics meet or exceed their respective thresholds, otherwise False. """
    for metric, threshold in thresholds.items():
        if metric not in metrics:
            raise ValueError(f"Metric '{metric}' is not computed. Available metrics: {list(metrics.keys())}")
        if metrics[metric] < threshold:
            return False
    return True