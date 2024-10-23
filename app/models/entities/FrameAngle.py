from sqlalchemy import Boolean, Column, Integer, JSON, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from app.config.database import Base  

class FrameAngle(Base):
    __tablename__ = 'frames_angles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    frame_number = Column(Integer, nullable=False)
    angles_results = Column(JSON, nullable=False)  # Salva il dict come JSON
    created_at = Column(TIMESTAMP, server_default=func.now())
    video_uuid = Column(UUID(as_uuid=True), nullable=False)
    keypoints = Column(JSON, nullable=False)
    is_last_frame = Column(Boolean, nullable=False, default=False)
