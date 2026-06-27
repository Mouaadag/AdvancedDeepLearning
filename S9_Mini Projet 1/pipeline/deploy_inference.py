"""Deploie inference_pipeline comme serveur HTTP via ZenML Pipeline Deployments.

Le modele est charge automatiquement depuis le ZenML Model Control Plane :
aucune intervention manuelle ni passage de chemin de fichier requis.

Usage:
    # Apres un run du pipeline d'entrainement
    uv run deploy_inference.py

    # Avec un nom de deploiement personnalise
    uv run deploy_inference.py --name mon-api

    # Tester l'API une fois deployee
    zenml deployment invoke sentiment-api --text="This movie is amazing!"

    # Gestion
    zenml deployment list
    zenml deployment delete sentiment-api
"""
import argparse
import logging

from inference_pipeline import inference_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploiement du modele de sentiment")
    parser.add_argument("--name", type=str, default="sentiment-api")
    args = parser.parse_args()

    # Le pipeline recupere automatiquement le dernier modele depuis le MCP ZenML.
    # Aucun on_init_kwargs necessaire : Model(version=ModelStages.LATEST) s'en charge.
    deployment = inference_pipeline.deploy(deployment_name=args.name)

    logger.info("API disponible : %s", deployment.url)
    logger.info("Invoquer  : zenml deployment invoke %s --text='Great film!'", args.name)
    logger.info("Lister    : zenml deployment list")
    logger.info("Arreter   : zenml deployment delete %s", args.name)


if __name__ == "__main__":
    main()
