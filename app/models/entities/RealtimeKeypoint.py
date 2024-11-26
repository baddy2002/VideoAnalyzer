from sqlalchemy import Boolean, Column, Integer, JSON, TIMESTAMP, String, Double
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from sqlalchemy.future import select
from app.config.database import Base, get_session  
import logging

logger = logging.getLogger(__name__)
class RealtimeKeypoint(Base):
    __tablename__ = 'realtime_keypoints'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    elaboration_uuid = Column(UUID(as_uuid=True), nullable=False)
    keypoints = Column(JSON, nullable=False)
    frame_number = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

    @staticmethod
    async def get_previous_keypoints(elaboration_uuid, frame_number, window_size):
        """
        Recupera i keypoints degli ultimi `window_size` frame precedenti al frame corrente.
        """
        async with get_session() as session:
            try:
                
                query = (
                    select(RealtimeKeypoint)
                    .where(RealtimeKeypoint.elaboration_uuid == elaboration_uuid, RealtimeKeypoint.frame_number <= frame_number)
                    .order_by(RealtimeKeypoint.frame_number.desc(), RealtimeKeypoint.created_at.desc())
                    .limit(window_size)
                )
                result = await session.execute(query)
                return result.scalars().all()
            except Exception as e:
                await session.rollback()  # Rollback in caso di errori
                logger.error(f"Errore durante il salvataggio dei keypoints realtime a db: {e}")
                raise