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
    data_dir:str = None
):
    if data_dir != None:
        return f.training(data_dir)
    else:
        raise HTTPException(404,"data_dir not found")

@router.post(path="/predict")
def predict(
    data:UploadFile = File(...),
    train_name:str = None
):
    img = Image.open(data.file)
    img.save("./predict_ds/"+data.filename)
    result = f.predict(data_name = data.filename, weight_dir=train_name)
    return{"result":result}