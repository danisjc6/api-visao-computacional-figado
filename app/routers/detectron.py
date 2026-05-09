from fastapi import APIRouter, UploadFile, File, Request
from app.utils import read_image, save_annotated_image
import numpy as np

router = APIRouter()

@router.post("/predict_auto")
async def predict_auto(request: Request, file: UploadFile = File(...)):
    classifier = request.app.state.classifier
    predictors = request.app.state.predictors
    class_names = request.app.state.class_names
    valid_liver_classes = request.app.state.valid_liver_classes

    # Ler imagem
    image = read_image(await file.read())

    # Classificar espécie
    especie, conf = classifier.predict(image)

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
    detected_names = [class_names[especie][int(c)] for c in pred_classes]
    if not any(name in valid_liver_classes[especie] for name in detected_names):
        return {"status": "rejeitado", "motivo": "Imagem não contém fígado"}

    # Preparar detecções para retorno
    detections = []
    for box, score, cls in zip(instances.pred_boxes.tensor, instances.scores, instances.pred_classes):
        cls_id = int(cls)
        detections.append({
            "classe": class_names[especie].get(cls_id, "desconhecida"),
            "score": round(float(score),3),
            "bbox": [int(v) for v in box.tolist()]
        })

    # Salvar imagem anotada
    image_path = save_annotated_image(image=image, instances=instances, especie=especie, class_names=class_names)

    return {
        "status": "ok",
        "especie": especie,
        "confidence_especie": round(conf,3),
        "num_instancias": len(detections),
        "deteccoes": detections,
        "imagem_anotada": image_path
    }
