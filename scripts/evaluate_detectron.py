import argparse
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


# ======================
# ARGUMENTOS CLI
# ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação Detectron2 (COCO) para fígado canino/felino"
    )
    parser.add_argument(
        "--species",
        required=True,
        choices=["canino", "felino"],
        help="Espécie do modelo"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Split do dataset para avaliação"
    )
    return parser.parse_args()


# ======================
# MAIN
# ======================
def main():
    args = parse_args()

    SPECIES = args.species
    SPLIT = args.split

    BASE_DATASET = "/home/daniela/Documentos/projeto RP treinamento/projeto RP/dataset_detectron"

    DATASET_NAME = f"figado_{SPECIES}_{SPLIT}"
    DATASET_IMG = f"{BASE_DATASET}/{SPECIES}/{SPLIT}/images"
    DATASET_ANN = f"{BASE_DATASET}/{SPECIES}/{SPLIT}/annotations.json"

    CONFIG_PATH = f"models/detectron/{SPECIES}/inferencia_{SPECIES}.yaml"
    WEIGHTS_PATH = f"models/detectron/{SPECIES}/model_final_{SPECIES}.pth"

    OUTPUT_DIR = f"results/eval/{SPECIES}/{SPLIT}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ======================
    # VALIDAÇÃO DE CAMINHOS
    # ======================
    for path in [DATASET_IMG, DATASET_ANN, CONFIG_PATH, WEIGHTS_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Caminho não encontrado: {path}")

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
    # CONFIGURAÇÃO DO MODELO
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

    print("\n✅ Avaliação concluída")
    print(f"Espécie: {SPECIES}")
    print(f"Split: {SPLIT}")
    print(results)


if __name__ == "__main__":
    main()
