from sqlalchemy import Column, Integer, JSON, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from app.config.database import Base  

class ElaborationFrames(Base):
    __tablename__ = 'frame_confrontations'

    uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    frame_number = Column(Integer, nullable=False)
    keypoints = Column(JSON, nullable=False)
    connections = Column(JSON, nullable=False)  
    elaboration_uuid = Column(UUID(as_uuid=True), nullable=False)       
    correct_keypoints = Column(JSON, nullable=False)