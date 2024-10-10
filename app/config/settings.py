import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import quote_plus

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


class Settings(BaseSettings):

    # App
    APP_NAME:  str = os.environ.get("APP_NAME", "FastAPI")   
    DEBUG: bool = bool(os.environ.get("DEBUG", False))
    
    # FrontEnd Application
    FRONTEND_HOST: str = os.environ.get("FRONTEND_HOST", "http://localhost:3000")       

    # POSTGRES Config
    POSTGRES_HOST: str = os.environ.get("POSTGRES_HOST", "postgres-service")
    POSTGRES_PORT: int = int(os.environ.get("POSTGRES_PORT", 5432))
    POSTGRES_USER: str = os.environ.get("POSTGRES_USER", 'image_analysis')
    POSTGRES_PASSWORD: str = os.environ.get("POSTGRES_PASSWORD", 'image_analysis')
    POSTGRES_DB: str = os.environ.get("POSTGRES_DB", 'image_analysis')
    DATABASE_URI: str =  f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    STATIC_FOLDER: str = os.environ.get("STATIC_FOLDER", "./static")  
    VIDEO_FOLDER: str = os.environ.get("VIDEO_FOLDER", "./static/video") 
    TEMPLATES_FOLDER: str = os.environ.get("TEMPLATES_FOLDER", "./templates")  