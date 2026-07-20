from fastapi import APIRouter, UploadFile, File, Request

from app.utils import read_image
from app.services.prediction_service import predict_liver

router = APIRouter()


@router.post("/predict_auto")
async def predict_auto(
    request: Request,
    file: UploadFile = File(...)
):
    image = read_image(await file.read())

    result = predict_liver(
        image=image,
        classifier=request.app.state.classifier,
        predictors=request.app.state.predictors,
        class_names=request.app.state.class_names,
        valid_liver_classes=request.app.state.valid_liver_classes
    )

    return result
