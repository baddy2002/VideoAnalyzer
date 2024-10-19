import logging

from sqlalchemy import UUID
from sqlalchemy.future import select

from config.settings import Settings
from app.config.database import get_session
from app.models.entities import FrameAngle, Video

logger = logging.getLogger(__name__)
settings = Settings()


#<======================================DB save angles======================================>

async def save_pose_angles_to_db(frame_number, angles_results, video_uuid, keypoints):
    angles_results_str_keys = {str(key): value for key, value in angles_results.items()}
    async with get_session() as session:
        try:
            pose_angle = FrameAngle.FrameAngle(
                frame_number=frame_number,
                angles_results=angles_results_str_keys,  # Salva tutto il dizionario come JSON
                video_uuid=video_uuid,
                keypoints=keypoints
            )
            session.add(pose_angle)
            await session.commit()
        except Exception as e:
            await session.rollback()  # Rollback in caso di errori
            logger.error(f"Errore durante il salvataggio degli angoli a db: {e}")
            raise

async def save_video_metadata(video_data):
    async with get_session() as session:
        try:
            video_entry = Video.Video(
                uuid=video_data['uuid'],
                name=video_data['name'],
                format=video_data['format'],
                size=video_data['size'],
                area=video_data['area'],
                portions=video_data['portions'],
                description=video_data['description'],
                fps=video_data['fps'],
                width=video_data['width'],
                height=video_data['height'],
            )
            session.add(video_entry)
            await session.commit()  # Usa await per il commit asincrono
        except Exception as e:
            await session.rollback()  # Rollback in caso di errori
            logger.error(f"Errore durante il salvataggio dei metadati video: {e}")
            raise


async def update_video_metadata(uuid: UUID, video_data: dict):
    async with get_session() as session:
        try:
            # Trova il video esistente con l'UUID fornito
            result = await session.execute(select(Video.Video).where(Video.Video.uuid == uuid))
            video = result.scalar_one_or_none()

            if video is None:
                raise ValueError(f'Video con uuid: {uuid} non trovato.')

            # Aggiorna i metadati del video
            for key, value in video_data.items():
                if hasattr(video, key):
                    setattr(video, key, value)

            # Salva le modifiche nel database
            await session.commit()  # Usa await per il commit asincrono
        except Exception as e:
            await session.rollback()  # Rollback in caso di errori
            logger.error(f"Errore durante l'aggiornamento dei metadati video: {e}")
            raise