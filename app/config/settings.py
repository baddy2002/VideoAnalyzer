import os
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
from typing import List

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):

    # App
    APP_NAME:  str = os.environ.get("APP_NAME", "FastAPI")   
    DEBUG: bool = bool(os.environ.get("DEBUG", False))
    
    # FrontEnd Application
    #FRONTEND_HOSTS: List[AnyHttpUrl] = [host for host in os.environ.get("FRONTEND_HOSTS", "http://localhost:4200").split(",") if host]
    FRONTEND_HOSTS: str= Field([host for host in os.environ.get("FRONTEND_HOSTS", "http://localhost:4200").split(",") if host])
    # POSTGRES Config
    POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "postgres-service")
    POSTGRES_PORT: int = int(os.environ.get("POSTGRES_PORT", 5432))
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", 'video_analysis')
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", 'video_analysis')
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", 'video_analysis')
    DATABASE_URI: str =  f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    STATIC_FOLDER: str = os.environ.get("STATIC_FOLDER", "./static")  
    VIDEO_FOLDER: str = os.environ.get("VIDEO_FOLDER", "./static/video") 
    TEMPLATES_FOLDER: str = os.environ.get("TEMPLATES_FOLDER", "./templates")  