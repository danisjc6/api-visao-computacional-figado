import os
import torch
import json
from PIL import Image
from sklearn.metrics import classification_report
from app.classifier import SpeciesClassifier

# ======================
# CONFIGURAÇÕES
# ======================
DATASET_DIR = "dataset_figado/test"
MODEL = "models/classifier/species_classifier.pth"
DEVICE = "cpu"

classifier = SpeciesClassifier(MODEL, threshold=0.0)

y_true = []
y_pred = []

for label in ["canino", "felino"]:
    folder = os.path.join(DATASET, label)
    for img_name in os.listdir(folder):
        img = Image.open(os.path.join(folder, img_name)).convert("RGB")
        pred, _ = classifier.predict(img)

        y_true.append(label)
        y_pred.append(pred if pred else "desconhecido")

report = classification_report(
    y_true,
    y_pred,
    labels=["canino", "felino"],
    output_dict=True
)

with open("results/metrics_classifier.json", "w") as f:
    json.dump(report, f, indent=2)

print("✅ Métricas do classificador salvas")