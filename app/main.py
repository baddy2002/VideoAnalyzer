from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from app.services.rs import imageAnalyzerServiceRs, videoAnalyzerServiceRs
from app.models.entities import FrameAngle, Video                                              #Sono da importare o non verranno aggiunte al db automaticamente
from app.config.database import async_engine, Base
from app.config.settings import Settings



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = Settings()

async def create_tables():
    async with async_engine.begin() as conn:  # Usa async_engine qui
        await conn.run_sync(Base.metadata.create_all)  # Crea le tabelle in modo asincrono

async def startup_event():
    # Crea le tabelle all'avvio
    await create_tables()

def create_application():
    application = FastAPI()
    application.add_middleware(
        CORSMiddleware,
        #allow_origins=settings.FRONTEND_HOSTS,  # domini consentiti
        allow_origins=["*"],  # o il tuo dominio specifico
        allow_credentials=True,
        allow_methods=["*"],  # consente tutti i metodi (GET, POST, ecc.)
        allow_headers=["*"],  # consente tutti gli header
    )
    application.include_router(imageAnalyzerServiceRs.image_analysis_router)
    application.include_router(videoAnalyzerServiceRs.video_analysis_router)
    #Crea tabelle e collega db
    try:
        application.add_event_handler("startup", startup_event)
        logger.info("Database connection successful!")
        logger.info("Tabelle create con successo!")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")

    return application



app = create_application()




