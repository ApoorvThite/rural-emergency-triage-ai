"""
Evaluation metrics for medical imaging tasks
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary',
) -> Dict[str, float]:
    """
    Compute classification metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC-ROC)
        average: Averaging strategy for multi-class
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0) * 100
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0) * 100
    metrics['sensitivity'] = metrics['recall']  # Same as recall
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0) * 100
    
    # Confusion matrix for binary classification
    if average == 'binary':
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
    
    # AUC-ROC
    if y_prob is not None:
        try:
            if average == 'binary':
                # Binary classification: use probability of positive class
                if y_prob.ndim == 2:
                    y_prob = y_prob[:, 1]
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            else:
                # Multi-class: use one-vs-rest
                metrics['auc_roc'] = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class='ovr',
                    average=average,
                )
        except ValueError as e:
            print(f"Warning: Could not compute AUC-ROC: {e}")
            metrics['auc_roc'] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
):
    """Plot ROC curves for each class"""
    
    plt.figure(figsize=(10, 8))
    
    if len(class_names) == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    else:
        # Multi-class classification
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            auc = roc_auc_score(y_true_binary, y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_confidence_intervals(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, tuple]:
    """Calculate confidence intervals using bootstrap"""
    
    from scipy import stats
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    
    bootstrap_metrics = {key: [] for key in metrics.keys()}
    
    for _ in range(n_bootstraps):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metrics
        boot_metrics = compute_classification_metrics(y_true_boot, y_pred_boot)
        
        for key in metrics.keys():
            if key in boot_metrics:
                bootstrap_metrics[key].append(boot_metrics[key])
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for key, values in bootstrap_metrics.items():
        lower = np.percentile(values, 100 * alpha / 2)
        upper = np.percentile(values, 100 * (1 - alpha / 2))
        confidence_intervals[key] = (lower, upper)
    
    return confidence_intervals
