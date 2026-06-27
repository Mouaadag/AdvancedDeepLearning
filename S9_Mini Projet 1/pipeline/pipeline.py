from zenml import Model, pipeline

from steps.evaluate import evaluate_model
from steps.ingest import load_imdb
from steps.preprocess import tokenize_data
from steps.register import register_model
from steps.train import train_model


@pipeline(
    name="sentiment_analysis_pipeline",
    model=Model(name="sentiment-distilbert-imdb"),
)
def sentiment_pipeline(
    max_train_samples: int = 2000,
    max_test_samples: int = 500,
    max_length: int = 128,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 16,
) -> None:
    train_df, test_df = load_imdb(
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )
    train_tokens, test_tokens = tokenize_data(
        train_df=train_df,
        test_df=test_df,
        max_length=max_length,
    )
    model_path = train_model(
        train_tokens=train_tokens,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )
    metrics = evaluate_model(
        model_path=model_path,
        test_tokens=test_tokens,
        batch_size=32,
    )
    register_model(model_path=model_path, metrics=metrics)
