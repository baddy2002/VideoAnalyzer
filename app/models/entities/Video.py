from sqlalchemy import Column, String, BigInteger, TIMESTAMP, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from app.config.database import Base  

class Video(Base):
    __tablename__ = 'videos'

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    format = Column(String, nullable=False)
    size = Column(BigInteger, nullable=False)  
    description = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    area = Column(JSON, nullable=False)  # Salva il dict come JSON
    portions = Column(JSON, nullable=False)  # Salva il dict come JSON
    thumbnail = Column(Text, nullable=True)

