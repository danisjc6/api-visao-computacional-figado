import os
import argparse
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import json
import cv2

# ======================================================
# FUNÇÃO: carregar LabelMe
# ======================================================
def load_labelme(dataset_dir):
    dataset_dicts = []

    images_dir = os.path.join(dataset_dir, "images")
    ann_dir = os.path.join(dataset_dir, "annotations")

    for ann_file in os.listdir(ann_dir):
        if not ann_file.endswith(".json"):
            continue

        json_path = os.path.join(ann_dir, ann_file)
        with open(json_path) as f:
            data = json.load(f)

        # 🔐 Usa apenas o nome do arquivo da imagem
        image_filename = os.path.basename(data["imagePath"])
        img_path = os.path.join(images_dir, image_filename)

        img = cv2.imread(img_path)

        # ❗ Proteção contra erro silencioso
        if img is None:
            print(f"⚠️ Imagem não encontrada: {img_path}")
            continue

        height, width = img.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": ann_file,
            "height": height,
            "width": width,
            "annotations": []
        }

        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]

            if label not in CLASSES:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            bbox = [
                min(xs),
                min(ys),
                max(xs) - min(xs),
                max(ys) - min(ys)
            ]

            record["annotations"].append({
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": CLASSES.index(label)
            })

        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)

    return dataset_dicts



# ======================================================
# AVALIAÇÃO POR ESPÉCIE
# ======================================================
def evaluate_species(
    species_name,
    dataset_dir,
    classes,
    config_path,
    weights_path,
    output_dir
):
    global CLASSES
    CLASSES = classes

    dataset_name = f"figado_{species_name}_val"

    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name,
            lambda: load_labelme(dataset_dir)
        )
        MetadataCatalog.get(dataset_name).thing_classes = classes

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = "cpu"

    cfg.DATASETS.TEST = (dataset_name,)

    evaluator = COCOEvaluator(
        dataset_name,
        cfg,
        False,
        output_dir=output_dir
    )

    loader = build_detection_test_loader(cfg, dataset_name)

    results = inference_on_dataset(
        DefaultPredictor(cfg).model,
        loader,
        evaluator
    )

    print(f"\n📊 RESULTADOS — {species_name.upper()}")
    print(results)

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Avalia modelos Detectron2 usando anotacoes LabelMe"
    )
    parser.add_argument("--dataset-dir", default="dataset_detectron")
    parser.add_argument("--models-dir", default="models/detectron")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    args = parse_args()

    os.makedirs(os.path.join(args.output_dir, "canino"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "felino"), exist_ok=True)

    # 🐶 CANINO
    evaluate_species(
        species_name="canino",
        dataset_dir=os.path.join(args.dataset_dir, "canino", "val"),
        classes=["figado_cao", "processo_papilar_canino"],
        config_path=os.path.join(args.models_dir, "canino", "inferencia_canino.yaml"),
        weights_path=os.path.join(args.models_dir, "canino", "model_final_canino.pth"),
        output_dir=os.path.join(args.output_dir, "canino"),
    )

    # 🐱 FELINO
    evaluate_species(
        species_name="felino",
        dataset_dir=os.path.join(args.dataset_dir, "felino", "val"),
        classes=["figado_felino", "processo_papilar_felino"],
        config_path=os.path.join(args.models_dir, "felino", "inferencia_felino.yaml"),
        weights_path=os.path.join(args.models_dir, "felino", "model_final_felino.pth"),
        output_dir=os.path.join(args.output_dir, "felino"),
    )
