from PIL import Image
import io
import numpy as np
import torch
from torchvision import transforms


def read_image(file_bytes):
    """
    Lê bytes enviados pela API e converte em imagem PIL RGB
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return image


def preprocess_classifier(image):
    """
    Pré-processamento para o classificador de espécie
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

import cv2
import os


def save_annotated_image(
    image,
    instances,
    especie,
    class_names,
    output_dir="outputs"
):
    """
    Salva imagem anotada com bounding boxes e labels
    """
    img = np.array(image).copy()

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    save_dir = os.path.join(output_dir, especie)
    os.makedirs(save_dir, exist_ok=True)

    for box, score, cls in zip(boxes, scores, classes):
        cls_id = int(cls)
        label = class_names[especie].get(cls_id, "desconhecida")
        text = f"{label} {score:.2f}"

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    filename = f"resultado_{os.getpid()}_{np.random.randint(10000)}.jpg"
    path = os.path.join(save_dir, filename)

    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return path

# ---- logging utilities
import json
import csv
from datetime import datetime
import os


LOG_DIR = "logs"
CSV_PATH = os.path.join(LOG_DIR, "predictions.csv")
JSONL_PATH = os.path.join(LOG_DIR, "predictions.jsonl")


def log_prediction(data: dict):
    os.makedirs(LOG_DIR, exist_ok=True)

    data["timestamp"] = datetime.now().isoformat()

    # ---- JSONL (1 linha = 1 predição)
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ---- CSV (resumo)
    csv_row = {
        "timestamp": data["timestamp"],
        "status": data.get("status"),
        "especie": data.get("especie"),
        "confidence_especie": data.get("confidence_especie"),
        "num_instancias": data.get("num_instancias"),
        "motivo": data.get("motivo"),
    }

    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
