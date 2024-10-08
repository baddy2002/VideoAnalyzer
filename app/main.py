from fastapi import FastAPI
from app.services.rs import imageAnalyzerServiceRs
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_application():
    application = FastAPI()
    application.include_router(imageAnalyzerServiceRs.image_analysis_router)
    return application


app = create_application()


@app.get("/")
async def root():
    return {"message": "Hi, I am Baddy. Awesome - Your setup is done & working, now you can use the app!"}


