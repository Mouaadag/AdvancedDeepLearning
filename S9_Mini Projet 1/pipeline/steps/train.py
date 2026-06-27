import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    get_linear_schedule_with_warmup,
)
from typing import Annotated

from zenml import step
from zenml.artifacts.artifact_config import ArtifactConfig

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
SAVE_DIR = "artifacts/distilbert_imdb"


def _make_tensor_dataset(df: pd.DataFrame) -> TensorDataset:
    input_ids = torch.tensor(np.array(df["input_ids"].tolist()), dtype=torch.long)
    attention_mask = torch.tensor(np.array(df["attention_mask"].tolist()), dtype=torch.long)
    labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, labels)


def _train_one_epoch(
    model: DistilBertForSequenceClassification,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids, mask, labels = (t.to(device) for t in batch)
        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


@step
def train_model(
    train_tokens: pd.DataFrame,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
) -> Annotated[str, ArtifactConfig(name="distilbert_checkpoint")]:
    """Fine-tune DistilBERT sur IMDb et logue les hyperparametres et la loss dans MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | Samples: %d", device, len(train_tokens))

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    dataset = _make_tensor_dataset(train_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    with mlflow.start_run(run_name="distilbert-train", nested=True):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "train_samples": len(train_tokens),
            }
        )
        for epoch in range(epochs):
            avg_loss = _train_one_epoch(model, loader, optimizer, scheduler, device)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            logger.info("Epoch %d/%d | loss: %.4f", epoch + 1, epochs, avg_loss)

        save_path = Path(SAVE_DIR)
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path)
        DistilBertTokenizer.from_pretrained(MODEL_NAME).save_pretrained(save_path)
        mlflow.log_artifacts(str(save_path), artifact_path="model")

    return str(save_path)
