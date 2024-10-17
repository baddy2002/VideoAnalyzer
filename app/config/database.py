from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from app.config.settings import Settings
import logging

settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Connettendo a : " + str(settings.DATABASE_URI))

# Crea il motore asincrono
async_engine = create_async_engine(settings.DATABASE_URI, 
                              pool_pre_ping=True,
                              pool_recycle=3600,
                              pool_size=20,
                              max_overflow=0,
                              echo=True)  # echo=True per logging delle query

# Crea la sessione asincrona
AsyncSessionLocal = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Funzione per ottenere una sessione asincrona
@asynccontextmanager
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session