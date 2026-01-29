import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report
)

@dataclass
class ModelMetrics:
    """Stores overall model performance metrics."""
    model_name: str
    accuracy: float
    auc_score: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    balanced_accuracy: float
    log_loss: float
    mcc: float
    kappa: float


def compute_metrics_multiclass(model_name: str, y_true, y_pred, y_pred_prob=None) -> tuple[ModelMetrics, pd.DataFrame]:
    """
    Compute overall + per-class metrics for multiclass classification.
    
    Returns:
        ModelMetrics (dataclass)
        per_class_df (pd.DataFrame)
    """

    # === Overall metrics ===
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Macro and weighted averages
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # AUC and LogLoss (only if probabilities provided)
    if y_pred_prob is not None:
        auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
        ll = log_loss(y_true, y_pred_prob)
    else:
        auc, ll = np.nan, np.nan

    overall_metrics = ModelMetrics(
        model_name=model_name,
        accuracy=accuracy,
        auc_score=auc,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        balanced_accuracy=balanced_acc,
        log_loss=ll,
        mcc=mcc,
        kappa=kappa
    )

    # === Per-class metrics ===
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class_df = pd.DataFrame(report_dict).transpose().reset_index()
    per_class_df.rename(columns={'index': 'class'}, inplace=True)

    # Add model name column
    per_class_df.insert(0, 'model_name', model_name)

    return overall_metrics, per_class_df


def save_metrics(overall_metrics: ModelMetrics, per_class_df: pd.DataFrame, base_path: str):
    """
    Saves overall and per-class metrics as CSVs.
    """
    overall_path = f"{base_path}/overall_metrics.csv"
    per_class_path = f"{base_path}/per_class_metrics.csv"

    # Append overall metrics
    try:
        df_overall = pd.read_csv(overall_path)
    except FileNotFoundError:
        df_overall = pd.DataFrame(columns=asdict(overall_metrics).keys())
    df_overall = pd.concat([df_overall, pd.DataFrame([asdict(overall_metrics)])], ignore_index=True)
    df_overall.to_csv(overall_path, index=False)

    # Append per-class metrics
    try:
        df_perclass = pd.read_csv(per_class_path)
    except FileNotFoundError:
        df_perclass = pd.DataFrame(columns=per_class_df.columns)
    df_perclass = pd.concat([df_perclass, per_class_df], ignore_index=True)
    df_perclass.to_csv(per_class_path, index=False)

    print(f"Saved overall metrics to: {overall_path}")
    print(f"Saved per-class metrics to: {per_class_path}")
