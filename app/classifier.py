import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image


class SpeciesClassifier:

    def __init__(self, model_path: str, threshold=0.7):

        self.threshold = threshold

        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        full_model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {full_model_path}"
            )

        self.device = "cpu"

        # ======================
        # RESNET18
        # ======================
        self.model = models.resnet18(weights=None)

        # saída binária
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            2
        )

        self.model = self.model.to(self.device)

        # ======================
        # LOAD PESOS
        # ======================
        state_dict = torch.load(
            full_model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)

        self.model.eval()

        # ======================
        # TRANSFORMS
        # ======================
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def predict(self, image: Image.Image):

        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)

        if conf.item() < self.threshold:
            return None, conf.item()

        species = "canino" if idx.item() == 0 else "felino"

        return species, conf.item()