"""
Avaliação oficial COCO
Gera métricas AP, AP50, AP75 etc.
"""
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# ======================
# CONFIGURAÇÕES GERAIS
# ======================

SPECIES = "canino"  # "canino" ou "felino"

BASE_DATASET = "/home/daniela/Documentos/projeto RP treinamento/projeto RP/dataset_detectron"

DATASET_NAME = f"figado_{SPECIES}_val"

DATASET_IMG = f"{BASE_DATASET}/{SPECIES}/val/images"
DATASET_ANN = f"{BASE_DATASET}/{SPECIES}/val/annotations.json"

CONFIG_PATH = f"models/detectron/{SPECIES}/inferencia_{SPECIES}.yaml"
WEIGHTS_PATH = f"models/detectron/{SPECIES}/model_final_{SPECIES}.pth"

OUTPUT_DIR = f"results/eval_{SPECIES}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# REGISTRO DO DATASET
# ======================

if DATASET_NAME not in DatasetCatalog.list():
    register_coco_instances(
        DATASET_NAME,
        {},
        DATASET_ANN,
        DATASET_IMG
    )

# ======================
# CONFIG DO MODELO
# ======================

cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
cfg.MODEL.DEVICE = "cpu"

# ⚠️ IMPORTANTE: dataset de teste
cfg.DATASETS.TEST = (DATASET_NAME,)

# ======================
# AVALIAÇÃO COCO
# ======================

predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator(
    DATASET_NAME,
    cfg,
    distributed=False,
    output_dir=OUTPUT_DIR
)

loader = build_detection_test_loader(cfg, DATASET_NAME)

results = inference_on_dataset(
    predictor.model,
    loader,
    evaluator
)

# ======================
# RESULTADOS
# ======================

print("\n📊 RESULTADOS —", SPECIES.upper())
print(results)
print(f"\n✔ Avaliação concluída. Resultados salvos em: {OUTPUT_DIR}")
