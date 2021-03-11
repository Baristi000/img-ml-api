from fastapi import APIRouter

from api import learn_img

api_router = APIRouter()

api_router.include_router(learn_img.router,prefix='/data',tags=['DATA'])
