from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import os


BASE_PATH = "models/detectron"

def load_predictor(species: str):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(BASE_PATH, species, f"inferencia_{species}.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(BASE_PATH, species, f"model_final_{species}.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return DefaultPredictor(cfg)


# Inicializa os dois modelos uma vez
predictors = {
    "canino": load_predictor("canino"),
    "felino": load_predictor("felino")
}


def predict_image_auto(image_path: str):
    """
    Recebe imagem, roda os dois modelos e retorna espécie + detecção com maior confiança
    """
    img = cv2.imread(image_path)
    results = {}
    
    for species, predictor in predictors.items():
        output = predictor(img)
        # Pega o score médio da predição (ou o maior score)
        if len(output["instances"].scores) > 0:
            max_score = output["instances"].scores.max().item()
        else:
            max_score = 0.0
        results[species] = {"output": output, "score": max_score}

    # Escolhe a espécie com maior score
    best_species = max(results, key=lambda k: results[k]["score"])
    best_output = results[best_species]["output"]

    return {
        "species": best_species,
        "score": results[best_species]["score"],
        "detections": best_output.to("cpu").__dict__  # ou formatar bbox, classe, score
    }
