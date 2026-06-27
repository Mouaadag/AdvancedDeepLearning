import logging
import sys
import types
from typing import Dict

# mlflow.transformers.save_model calls get_default_pip_requirements unconditionally
# before checking user-supplied pip_requirements — it crashes when torchvision is absent.
# Inject a stub so the version probe succeeds; our explicit pip_requirements below
# will replace the auto-detected list in the final MLmodel artifact.
if "torchvision" not in sys.modules:
    _stub = types.ModuleType("torchvision")
    _stub.__version__ = "0.0.0"
    sys.modules["torchvision"] = _stub

import mlflow
import torch
import transformers
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from zenml import step

logger = logging.getLogger(__name__)

MIN_ACCURACY = 0.80
MODEL_REGISTRY_NAME = "sentiment-distilbert-imdb"

_PIP_REQUIREMENTS = [
    f"transformers=={transformers.__version__}",
    f"torch=={torch.__version__}",
    "tokenizers",
    "accelerate",
]


@step
def register_model(
    model_path: str,
    metrics: Dict[str, float],
) -> None:
    """Enregistre le modele dans MLflow Model Registry si accuracy >= seuil."""
    acc = metrics.get("accuracy", 0.0)

    if acc < MIN_ACCURACY:
        logger.warning(
            "Modele rejete : accuracy=%.4f < seuil=%.2f", acc, MIN_ACCURACY
        )
        return

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)

    with mlflow.start_run(run_name="distilbert-register") as run:
        mlflow.log_metrics(metrics)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            name="model",
            task="text-classification",
            pip_requirements=_PIP_REQUIREMENTS,
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

    logger.info(
        "Modele enregistre : %s v%s (accuracy=%.4f)",
        MODEL_REGISTRY_NAME,
        result.version,
        acc,
    )
