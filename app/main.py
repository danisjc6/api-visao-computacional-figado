from fastapi import FastAPI, UploadFile, File
import numpy as np
import yaml

from app.classifier import SpeciesClassifier
from app.detectron import load_predictor
from app.utils import read_image, preprocess_classifier, save_annotated_image
from app.routers import detectron

# ======================
# Instância única do FastAPI
# ======================
app = FastAPI(title="API Visão Computacional – Fígado")

# ======================
# Load CONFIG
# ======================
with open("configs/app.yaml") as f:
    CFG = yaml.safe_load(f)

# ======================
# Load MODELS (uma vez só)
# ======================
classifier = SpeciesClassifier(CFG["classifier"]["model"], threshold=0.7)

predictors = {
    "canino": load_predictor("canino"),
    "felino": load_predictor("felino")
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
# Endpoint principal
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # ---- read image
    image = read_image(await file.read())

    # ---- species classification
    especie, conf = classifier.predict(image)

    if especie is None:
        return {
            "status": "rejeitado",
            "motivo": "Imagem não parece ser fígado de cão ou gato",
            "confidence": round(conf, 3)
        }

    # ---- detectron inference
    predictor = predictors[especie]
    outputs = predictor(np.array(image))
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return {
            "status": "rejeitado",
            "motivo": "Nenhuma estrutura hepática detectada"
        }

    # ---- extrair boxes, scores e classes corretamente
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes

    # ---- validar classes detectadas
    detected_names = [CLASS_NAMES[especie][int(c)] for c in classes]

    if not any(name in VALID_LIVER_CLASSES[especie] for name in detected_names):
        return {
            "status": "rejeitado",
            "motivo": "Imagem não contém fígado"
        }

    # ---- criar lista de deteccoes
    detections = []
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
# Router detectron separado
# ======================
app.include_router(detectron.router, prefix="/detectron")

# ======================
# Health check
# ======================
@app.get("/")
def health():
    return {"status": "ok", "models": ["canino", "felino"]}
