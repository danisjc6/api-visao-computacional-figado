import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SpeciesClassifier:
    def __init__(self, model_path: str, threshold=0.7):
        self.threshold = threshold

        base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        full_model_path = os.path.join(base_dir, model_path)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {full_model_path}")

        self.device = "cpu"

        self.model = torch.load(
          full_model_path,
          map_location=self.device,
          weights_only=False
        )

        self.model.eval()

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
