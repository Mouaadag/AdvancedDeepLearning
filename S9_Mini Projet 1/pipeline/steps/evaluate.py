import logging
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification
from zenml import step

logger = logging.getLogger(__name__)


def _make_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    input_ids = torch.tensor(np.array(df["input_ids"].tolist()), dtype=torch.long)
    attention_mask = torch.tensor(np.array(df["attention_mask"].tolist()), dtype=torch.long)
    labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)


def _compute_metrics(
    labels: List[int], preds: List[int]
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary")),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


def _predict(
    model: DistilBertForSequenceClassification,
    test_tokens: pd.DataFrame,
    batch_size: int = 32,
) -> Tuple[List[int], List[int]]:
    device = next(model.parameters()).device
    dataset = _make_tensor_dataset(test_tokens)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids, mask, labels = (t.to(device) for t in batch)
            logits = model(input_ids=ids, attention_mask=mask).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels


@step
def evaluate_model(
    model_path: str,
    test_tokens: pd.DataFrame,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Evalue le modele sur le test set et logue les metriques dans MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

    preds, labels = _predict(model, test_tokens, batch_size=batch_size)
    metrics = _compute_metrics(labels, preds)

    with mlflow.start_run(run_name="distilbert-eval", nested=True):
        mlflow.log_metrics(metrics)

    logger.info("Metriques: %s", metrics)
    return metrics
