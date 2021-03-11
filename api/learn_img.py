from fastapi import APIRouter, Body, HTTPException, UploadFile, File
from typing import List
import functions.ml_function as f
from PIL import Image
from io import BytesIO

router = APIRouter()

@router.post('/upload')
def upload_data(
    train_data:UploadFile = File(...),
    folder_name:str = None
):
    data_dir = f.untar_file(train_data,folder_name)
    return{"data_dir":data_dir}

@router.get("/training")
def train_model(
    train_dir:str = None
):
    if train_dir != None:
        return f.training(traindir)
    else:
        raise HTTPException(404,"data_dir not found")

@router.post(path="/predict")
def predict(
    data:UploadFile = File(...),
    train_dir:str = None
):
    img = Image.open(data.file)
    img.save("./predict_ds/"+data.filename)
    result = f.predict(data_name = data.filename, weight_dir=train_dir)
    return{"result":result}

@router.get("/conts-train")
def const_train(
    train_dir: str = None,
    epochs:int = 10
):
    return f.conts_train(train_dir, epochs)