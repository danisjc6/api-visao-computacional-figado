import numpy as np

from app.utils import save_annotated_image


def _format_detections(instances, especie, class_names):
    detections = []
    for box, score, cls in zip(
        instances.pred_boxes.tensor,
        instances.scores,
        instances.pred_classes
    ):
        cls_id = int(cls)
        detections.append({
            "classe": class_names[especie].get(cls_id, "desconhecida"),
            "score": round(float(score), 3),
            "bbox": [int(v) for v in box.tolist()]
        })
    return detections


def _detect_with_species(image, especie, predictor):
    outputs = predictor(np.array(image))
    instances = outputs["instances"].to("cpu")
    score = float(instances.scores.max()) if len(instances) > 0 else 0.0
    return instances, score


def _choose_species_by_detection(image, predictors):
    best_species = None
    best_instances = None
    best_score = 0.0

    for especie, predictor in predictors.items():
        instances, score = _detect_with_species(image, especie, predictor)
        if score > best_score:
            best_species = especie
            best_instances = instances
            best_score = score

    return best_species, best_instances, best_score


def predict_liver(image, classifier, predictors, class_names, valid_liver_classes):
    if classifier is None:
        especie, instances, conf = _choose_species_by_detection(image, predictors)
        if especie is None or instances is None or len(instances) == 0:
            return {
                "status": "rejeitado",
                "motivo": "Nenhuma estrutura hepática detectada"
            }
    else:
        especie, conf = classifier.predict(image)
        if especie is None:
            return {
                "status": "rejeitado",
                "motivo": "Imagem não parece ser fígado de cão ou gato",
                "confidence": round(conf, 3)
            }

        instances, _ = _detect_with_species(image, especie, predictors[especie])
        if len(instances) == 0:
            return {
                "status": "rejeitado",
                "motivo": "Nenhuma estrutura hepática detectada"
            }

    detected_names = [
        class_names[especie][int(c)]
        for c in instances.pred_classes
    ]

    if not any(name in valid_liver_classes[especie] for name in detected_names):
        return {
            "status": "rejeitado",
            "motivo": "Imagem não contém fígado"
        }

    detections = _format_detections(instances, especie, class_names)
    image_path = save_annotated_image(
        image=image,
        instances=instances,
        especie=especie,
        class_names=class_names
    )

    return {
        "status": "ok",
        "especie": especie,
        "confidence_especie": round(conf, 3),
        "num_instancias": len(detections),
        "deteccoes": detections,
        "imagem_anotada": image_path
    }
