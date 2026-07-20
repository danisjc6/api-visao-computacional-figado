import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def load_predictor(species: str, model_config: dict):
    """
    Carrega um predictor do Detectron2 para a espécie especificada.
    """
    cfg = get_cfg()

    yaml_path = model_config["config"]
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"Config Detectron para '{species}' não encontrado: {yaml_path}"
        )

    cfg.merge_from_file(yaml_path)

    weights_path = model_config["weights"]
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Modelo Detectron para '{species}' não encontrado: {weights_path}"
        )

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_config.get("score_threshold", 0.35)
    return DefaultPredictor(cfg)
