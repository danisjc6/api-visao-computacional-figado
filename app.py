from fastapi import FastAPI, UploadFile, File
from app.classifier import SpeciesClassifier
from app.detectron import load_predictor
from app.utils import (
    read_image,
    preprocess_classifier,
    save_annotated_image,
    log_prediction
)
import torch
import yaml
import numpy as np
import os

app = FastAPI(title="API Fígado Canino/Felino")

# ======================
# LOAD CONFIG
# ======================
with open("configs/app.yaml") as f:
    CFG = yaml.safe_load(f)

# ======================
# LOAD MODELS (uma vez só)
# ======================
classifier = SpeciesClassifier(
    CFG["classifier"]["model"],
    threshold=0.7
)

# Apenas passamos a espécie; load_predictor já monta os caminhos
predictors = {
    "canino": load_predictor("canino")[0],  # [0] pega o DefaultPredictor
    "felino": load_predictor("felino")[0]
}

VALID_LIVER_CLASSES = {
    "canino": ["figado_cao", "processo_papilar_canino"],
    "felino": ["figado_felino", "processo_papilar_felino"]
}

CLASS_NAMES = {
    "canino": {
        0: "figado_cao",
        1: "processo_papilar_canino"
    },
    "felino": {
        0: "figado_felino",
        1: "processo_papilar_felino"
    }
}

# ======================
# ENDPOINT
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---- ler imagem
    image = read_image(await file.read())

    # ---- preprocess para classifier
    img_tensor = preprocess_classifier(image)

    # ---- classificação de espécie
    especie, conf = classifier.predict(img_tensor)

    if especie is None:
        return {
            "status": "rejeitado",
            "motivo": "Imagem não parece ser fígado de cão ou gato",
            "confidence": round(conf, 3)
        }

    # ---- inferência Detectron
    predictor = predictors[especie]
    outputs = predictor(np.array(image))
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return {
            "status": "rejeitado",
            "motivo": "Nenhuma estrutura hepática detectada"
        }

    # ---- validar classes detectadas
    pred_classes = instances.pred_classes.tolist()
    detected_names = [CLASS_NAMES[especie][int(c)] for c in pred_classes]

    if not any(name in VALID_LIVER_CLASSES[especie] for name in detected_names):
        return {
            "status": "rejeitado",
            "motivo": "Imagem não contém fígado"
        }

    # ---- extrair detecções
    detections = []
    boxes = instances.pred_boxes if hasattr(instances, "pred_boxes") else None
    scores = instances.scores if hasattr(instances, "scores") else None
    classes = instances.pred_classes if hasattr(instances, "pred_classes") else None

    for box, score, cls in zip(boxes, scores, classes):
        cls_id = int(cls)
        detections.append({
            "classe": CLASS_NAMES[especie].get(cls_id, "desconhecida"),
            "score": round(float(score), 3),
            "bbox": [int(v) for v in box.tolist()]
        })

    # ---- salvar imagem anotada
    image_path = save_annotated_image(
        image=image,
        instances=instances,
        especie=especie,
        class_names=CLASS_NAMES
    )

    return {
        "status": "ok",
        "especie": especie,
        "confidence_especie": round(conf, 3),
        "num_instancias": len(detections),
        "deteccoes": detections,
        "imagem_anotada": image_path
    }

# ======================
# Health check
# ======================
@app.get("/")
def health():
    return {"status": "ok", "models": ["canino", "felino"]}
