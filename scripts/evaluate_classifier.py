import argparse
import json
import os

from PIL import Image
from sklearn.metrics import classification_report

from app.classifier import SpeciesClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Avalia o classificador de especie")
    parser.add_argument("--dataset-dir", default="dataset_figado/test")
    parser.add_argument("--model", default="models/classifier/species_classifier.pth")
    parser.add_argument("--output", default="results/metrics_classifier.json")
    return parser.parse_args()


def main():
    args = parse_args()
    classifier = SpeciesClassifier(args.model, threshold=0.0)

    y_true = []
    y_pred = []

    for label in ["canino", "felino"]:
        folder = os.path.join(args.dataset_dir, label)
        for img_name in os.listdir(folder):
            img = Image.open(os.path.join(folder, img_name)).convert("RGB")
            pred, _ = classifier.predict(img)

            y_true.append(label)
            y_pred.append(pred if pred else "desconhecido")

    report = classification_report(
        y_true,
        y_pred,
        labels=["canino", "felino"],
        output_dict=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Metricas do classificador salvas em: {args.output}")


if __name__ == "__main__":
    main()
