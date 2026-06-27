"""Pipeline d'inference deploye comme serveur HTTP via ZenML Pipeline Deployments.

Le modele est charge automatiquement depuis le ZenML Model Control Plane
(derniere version du modele 'sentiment-distilbert-imdb') sans intervention manuelle.
"""
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from zenml import Model, pipeline
from zenml.enums import ModelStages

from steps.predict import predict


def _load_model() -> dict:
    """Charge le modele une seule fois au demarrage depuis le ZenML Model Control Plane."""
    zenml_model = Model(name="sentiment-distilbert-imdb", version=ModelStages.LATEST)
    model_path: str = zenml_model.get_artifact("distilbert_checkpoint").load()

    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


@pipeline(
    model=Model(name="sentiment-distilbert-imdb", version=ModelStages.LATEST),
    on_init=_load_model,
)
def inference_pipeline(text: str = "This movie is amazing!") -> dict:
    return predict(text=text)
