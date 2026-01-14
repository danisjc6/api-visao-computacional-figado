"""
Inferência simples com Detectron2
Uso: debug e visualização
"""


from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os

# ======================
# CONFIGURAÇÕES
# ======================
DATASET_NAME = "figado_test"
DATASET_IMG = "dataset_detectron/images"
DATASET_ANN = "dataset_detectron/annotations.json"

CONFIG_PATH = "models/detectron/canino/inferencia_canino.yaml"
WEIGHTS_PATH = "models/detectron/canino/model_final.pth"

OUTPUT_DIR = "scripts/results/detectron_canino"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# REGISTRO DATASET
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

# ======================
# AVALIAÇÃO
# ======================
evaluator = COCOEvaluator(
    DATASET_NAME,
    cfg,
    False,
    output_dir=OUTPUT_DIR
)

loader = build_detection_test_loader(cfg, DATASET_NAME)

results = inference_on_dataset(
    DefaultPredictor(cfg).model,
    loader,
    evaluator
)

print("✔ Avaliação concluída")
print(results)
