from fastapi import FastAPI, UploadFile, File
import yaml

from app.classifier import SpeciesClassifier
from app.detectron import load_predictor
from app.utils import read_image
from app.routers import detectron
from app.services.prediction_service import predict_liver

# ======================
# Instância única do FastAPI
# ======================
app = FastAPI(title="API Visão Computacional – Fígado")

# ======================
# Load CONFIG
# ======================
with open("configs/app.yaml") as f:
    CFG = yaml.safe_load(f)

# ======================
# Load MODELS (uma vez só)
# ======================
classifier = SpeciesClassifier(
    CFG["classifier"]["model"],
    threshold=CFG["classifier"].get("threshold", 0.7),
)

predictors = {
    "canino": load_predictor("canino", CFG["detectron"]["canino"]),
    "felino": load_predictor("felino", CFG["detectron"]["felino"])
}

CLASS_NAMES = CFG["detectron"]["class_names"]
VALID_LIVER_CLASSES = CFG["detectron"]["valid_liver_classes"]

app.state.classifier = classifier
app.state.predictors = predictors
app.state.class_names = CLASS_NAMES
app.state.valid_liver_classes = VALID_LIVER_CLASSES

# ======================
# Endpoint principal
# ======================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_image(await file.read())

    return predict_liver(
        image=image,
        classifier=classifier,
        predictors=predictors,
        class_names=CLASS_NAMES,
        valid_liver_classes=VALID_LIVER_CLASSES,
    )

# ======================
# Router detectron separado
# ======================
app.include_router(detectron.router, prefix="/detectron")

# ======================
# Health check
# ======================
@app.get("/")
def health():
    return {"status": "ok", "models": ["canino", "felino"]}
