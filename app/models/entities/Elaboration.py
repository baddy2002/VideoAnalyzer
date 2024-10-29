from sqlalchemy import Column, Double, Integer, String, BigInteger, TIMESTAMP, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from app.config.database import Base
from app.models.enums.ElaborationStatus import ElaborationStatus  

class Elaboration(Base):
    __tablename__ = 'elaborations'

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    format = Column(String, nullable=False)
    size = Column(BigInteger, nullable=False)  
    created_at = Column(TIMESTAMP, server_default=func.now())
    thumbnail = Column(Text, nullable=True)
    video_uuid = Column(String, nullable=False)
    status =  Column(String, nullable=False, default=str(ElaborationStatus.CREATED.value))
