import argparse
import cv2
import os

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo


def load_predictor(species):
    base_dir = f"models/detectron/{species}"

    config_path = os.path.join(base_dir, f"inferencia_{species}.yaml")
    weights_path = os.path.join(base_dir, f"model_final_{species}.pth")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def main(args):
    predictor, cfg = load_predictor(args.species)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Imagem não encontrada: {args.image}")

    outputs = predictor(image)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.0
    )

    out = v.draw_instance_predictions(
        outputs["instances"].to("cpu")
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out.get_image()[:, :, ::-1])

    print(f"✅ Predição salva em: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", required=True, choices=["canino", "felino"])
    parser.add_argument("--image", required=True, help="Caminho da imagem de entrada")
    parser.add_argument("--output", default="output/predicao.jpg")

    args = parser.parse_args()
    main(args)


