from fastapi import APIRouter, UploadFile, File
from app.services.detectron import load_predictor
from app.classifier import SpeciesClassifier
from app.utils import read_image, preprocess_classifier, save_annotated_image
import numpy as np

router = APIRouter()

# Carregar modelos uma vez
classifier = SpeciesClassifier(
    "models/classifier/species_classifier.pth",  # ajuste para seu caminho
    threshold=0.7
)

predictors = {
    "canino": load_predictor("canino"),
    "felino": load_predictor("felino")
}

CLASS_NAMES = {
    "canino": {0: "figado_cao", 1: "processo_papilar_canino"},
    "felino": {0: "figado_felino", 1: "processo_papilar_felino"}
}

VALID_LIVER_CLASSES = {
    "canino": ["figado_cao", "processo_papilar_canino"],
    "felino": ["figado_felino", "processo_papilar_felino"]
}

@router.post("/predict_auto")
async def predict_auto(file: UploadFile = File(...)):
    # Ler imagem
    image = read_image(await file.read())

    # Preprocessar para classificador
    img_tensor = preprocess_classifier(image)

    # Classificar espécie
    especie, conf = classifier.predict(img_tensor)

    if especie is None:
        return {"status": "rejeitado", "motivo": "Não é fígado de cão ou gato", "confidence": round(conf,3)}

    # Detectron
    predictor = predictors[especie]
    outputs = predictor(np.array(image))
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return {"status": "rejeitado", "motivo": "Nenhuma estrutura hepática detectada"}

    # Validar classes detectadas
    pred_classes = instances.pred_classes.tolist()
    detected_names = [CLASS_NAMES[especie][int(c)] for c in pred_classes]
    if not any(name in VALID_LIVER_CLASSES[especie] for name in detected_names):
        return {"status": "rejeitado", "motivo": "Imagem não contém fígado"}

    # Preparar detecções para retorno
    detections = []
    for box, score, cls in zip(instances.pred_boxes.tensor, instances.scores, instances.pred_classes):
        cls_id = int(cls)
        detections.append({
            "classe": CLASS_NAMES[especie].get(cls_id, "desconhecida"),
            "score": round(float(score),3),
            "bbox": [int(v) for v in box.tolist()]
        })

    # Salvar imagem anotada
    image_path = save_annotated_image(image=image, instances=instances, especie=especie, class_names=CLASS_NAMES)

    return {
        "status": "ok",
        "especie": especie,
        "confidence_especie": round(conf,3),
        "num_instancias": len(detections),
        "deteccoes": detections,
        "imagem_anotada": image_path
    }
