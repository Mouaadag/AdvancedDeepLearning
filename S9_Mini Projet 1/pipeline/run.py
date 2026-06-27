import argparse
import logging

import mlflow

from inference_pipeline import inference_pipeline
from pipeline import sentiment_pipeline

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline (ZenML + MLflow)")
    parser.add_argument("--max_train", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mlflow_uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--deploy", action="store_true", help="Deployer l'API apres l'entrainement")
    parser.add_argument("--deploy_name", type=str, default="sentiment-api")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("sentiment-analysis-imdb")

    run = sentiment_pipeline.with_options(enable_cache=not args.no_cache)
    run(
        max_train_samples=args.max_train,
        max_test_samples=args.max_test,
        max_length=args.max_length,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    if args.deploy:
        logger.info("Lancement du deploiement : %s", args.deploy_name)
        deployment = inference_pipeline.deploy(deployment_name=args.deploy_name)
        logger.info("API disponible : %s", deployment.url)
        logger.info("Test : zenml deployment invoke %s --text='Great film!'", args.deploy_name)


if __name__ == "__main__":
    main()
