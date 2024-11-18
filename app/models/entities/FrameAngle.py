from sqlalchemy import Boolean, Column, Integer, JSON, TIMESTAMP, String, Double
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
    min_x= Column(Double, nullable=False, default=1000)
    min_x_key = Column(Integer, nullable=False, default=0)
    min_y= Column(Double, nullable=False, default=1000)
    min_y_key = Column(Integer, nullable=False, default=0)
    max_x= Column(Double, nullable=False, default=0)
    max_x_key = Column(Integer, nullable=False, default=0)
    max_y= Column(Double, nullable=False, default=0)
    max_y_key = Column(Integer, nullable=False, default=0)
    barycenter_x = Column(Double, nullable=False, default=0)
    barycenter_y = Column(Double, nullable=False, default=0)

