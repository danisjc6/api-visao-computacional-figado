import numpy as np

from app.utils import (
    build_annotated_image_base64,
    log_prediction,
)


def predict_liver(
    image,
    classifier,
    predictors,
    class_names,
    valid_liver_classes,
):
    """
    Executa todo o pipeline de predição.

    Fluxo:
        1. Classifica a espécie (canino/felino)
        2. Executa o Detectron correspondente
        3. Valida se há estruturas detectadas
        4. Valida se as classes pertencem ao fígado
        5. Salva a imagem anotada
        6. Registra a predição em log
    """

    # ----------------------------
    # 1. Classificação da espécie
    # ----------------------------
    especie, conf = classifier.predict(image)

    if especie is None:

        result = {
            "status": "rejeitado",
            "motivo": "Não é fígado de cão ou gato",
            "confidence": round(conf, 3),
        }

        log_prediction(result)
        return result

    # ----------------------------
    # 2. Detectron
    # ----------------------------
    predictor = predictors[especie]

    image_np = np.array(image)

    outputs = predictor(image_np)

    instances = outputs["instances"].to("cpu")

    # ----------------------------
    # 3. Nenhuma estrutura encontrada
    # ----------------------------
    if len(instances) == 0:

        result = {
            "status": "rejeitado",
            "especie": especie,
            "confidence_especie": round(conf, 3),
            "motivo": "Nenhuma estrutura hepática detectada",
        }

        log_prediction(result)
        return result

    # ----------------------------
    # 4. Validar classes detectadas
    # ----------------------------
    pred_classes = instances.pred_classes.tolist()

    detected_names = [
        class_names[especie].get(int(cls), "desconhecida")
        for cls in pred_classes
    ]

    if not any(
        classe in valid_liver_classes[especie]
        for classe in detected_names
    ):

        result = {
            "status": "rejeitado",
            "especie": especie,
            "confidence_especie": round(conf, 3),
            "motivo": "Imagem não contém estruturas hepáticas válidas",
        }

        log_prediction(result)
        return result

    # ----------------------------
    # 5. Preparar detecções
    # ----------------------------
    detections = []

    for box, score, cls in zip(
        instances.pred_boxes.tensor,
        instances.scores,
        instances.pred_classes,
    ):

        detections.append({
            "classe": class_names[especie].get(
                int(cls),
                "desconhecida",
            ),
            "score": round(float(score), 3),
            "bbox": [int(v) for v in box.tolist()],
        })

    # ----------------------------
    # 6. Gera imagem anotada
    # ----------------------------
    imagem_base64 = build_annotated_image_base64(
        image=image,
        instances=instances,
        especie=especie,
        class_names=class_names,
    )

    # ----------------------------
    # 7. Resultado final
    # ----------------------------
    result = {
        "status": "ok",
        "especie": especie,
        "confidence_especie": round(conf, 3),
        "num_instancias": len(detections),
        "deteccoes": detections,
        "imagem_anotada": imagem_base64
    }

    log_prediction(result)

    return result
