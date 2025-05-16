import deepchem as dc
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_model(model, test_dataset):
    """
    Args:
        model (dc.models.GraphConvModel): The trained DeepChem model.
        test_dataset (dc.data.Dataset): The test dataset.
    Returns:
        dict: Dictionary with ROC AUC and accuracy.
    """
    # Get predictions
    y_pred = model.predict(test_dataset)
    y_true = test_dataset.y

    # For binary classification, get the positive class probabilities
    y_pred_prob = y_pred[:, 0]

    # Convert probabilities to class labels (0 or 1)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Compute metrics
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_class)

    return {
        "ROC AUC": roc_auc,
        "Accuracy": acc
    }
