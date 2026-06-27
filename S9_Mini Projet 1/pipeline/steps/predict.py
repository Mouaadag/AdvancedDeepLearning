import torch
from typing import Annotated

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from zenml import get_step_context, step

_LABELS = {0: "negative", 1: "positive"}


@step
def predict(text: str) -> Annotated[dict, "prediction"]:
    """Inference sur un texte en utilisant le modele charge par on_init."""
    state: dict = get_step_context().pipeline_state
    model: DistilBertForSequenceClassification = state["model"]
    tokenizer: DistilBertTokenizer = state["tokenizer"]

    enc = tokenizer(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits

    pred = logits.argmax(dim=-1).item()
    score = torch.softmax(logits, dim=-1)[0][pred].item()

    return {"text": text, "label": _LABELS[pred], "score": round(score, 4)}
