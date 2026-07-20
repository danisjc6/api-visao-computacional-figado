from PIL import Image
import io
import numpy as np
import cv2
import base64
import json
import csv
from datetime import datetime
import os

def read_image(file_bytes):
    """
    Lê bytes enviados pela API e converte em imagem PIL RGB
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return image


def build_annotated_image_base64(
    image,
    instances,
    especie,
    class_names,
):
    """
    Desenha as detecções e devolve a imagem em Base64.
    """

    img = np.array(image).copy()

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    for box, score, cls in zip(
        boxes,
        scores,
        classes,
    ):

        label = class_names[especie].get(
            int(cls),
            "desconhecida"
        )

        text = f"{label} {score:.2f}"

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img,
            text,
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    success, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
    )

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")


# ---- logging utilities

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
