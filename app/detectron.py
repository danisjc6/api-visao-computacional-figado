# api_visao_computacional/app/detectron.py
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# Base path onde estão os arquivos de configuração e pesos
BASE_PATH = "models/detectron"

def load_predictor(species: str):
    """
    Carrega um predictor do Detectron2 para a espécie especificada ('canino' ou 'felino').

    Args:
        species (str): 'canino' ou 'felino'

    Returns:
        DefaultPredictor: objeto para realizar inferência
    """
    cfg = get_cfg()

    # Caminho do YAML de inferência (configuração)
    yaml_path = os.path.join(BASE_PATH, species, f"inferencia_{species}.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file '{yaml_path}' não encontrado!")

    cfg.merge_from_file(yaml_path)

    # Caminho do modelo treinado
    weights_path = os.path.join(BASE_PATH, species, f"model_final_{species}.pth")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Modelo '{weights_path}' não encontrado!")

    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold de confiança
    return DefaultPredictor(cfg)
