import logging
import uuid

from sqlalchemy import UUID
from sqlalchemy.future import select

from app.models.enums.ElaborationStatus import ElaborationStatus
from config.settings import Settings
from app.config.database import get_session
from app.models.entities import Elaboration, FrameAngle, Video

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


async def update_frame_angle_metadata(video_id: UUID, frame_number: int, frame_data: dict):
    async with get_session() as session:
        try:
            # Trova il FrameAngle esistente con l'UUID fornito
            result = await session.execute(select(FrameAngle.FrameAngle).where(FrameAngle.FrameAngle.video_uuid == video_id, FrameAngle.FrameAngle.frame_number == frame_number))
            frame_angle = result.scalar_one_or_none()

            if frame_angle is None:
                raise ValueError(f'FrameAngle con uuid: {video_id} e frame {frame_number} non trovato.')

            # Aggiorna i metadati del frame_angle
            for key, value in frame_data.items():
                if hasattr(frame_angle, key):
                    setattr(frame_angle, key, value)

            # Salva le modifiche nel database
            await session.commit()  # Usa await per il commit asincrono
        except Exception as e:
            await session.rollback()  # Rollback in caso di errori
            logger.error(f"Errore durante l'aggiornamento dei metadati frame_angle: {e}")
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


async def save_elaboration(
    elaboration_uuid: str,
    name: str = None,
    format: str = "mp4",
    size: int = 0,
    thumbnail: str = None,
    video_uuid: str = None,
    status: ElaborationStatus = ElaborationStatus.CREATED
):
    async with get_session() as session:
        try:
            # Verifica se l'UUID è valido
            try:
                UUID(elaboration_uuid)  # Solleva ValueError se l'UUID non è valido
            except ValueError:
                raise ValueError(f"L'UUID fornito '{elaboration_uuid}' non è valido.")

            # Verifica se l'elaborazione esiste già nel database
            existing_elaboration = await session.execute(
                select(Elaboration.Elaboration).where(Elaboration.Elaboration.uuid == elaboration_uuid)
            )
            existing_elaboration = existing_elaboration.scalar()

            if existing_elaboration:
                # Se l'elaborazione esiste già, esegui un aggiornamento
                existing_elaboration.name = name or existing_elaboration.name
                existing_elaboration.format = format
                existing_elaboration.size = size
                existing_elaboration.thumbnail = thumbnail
                existing_elaboration.video_uuid = video_uuid or existing_elaboration.video_uuid
                existing_elaboration.status = status.value

                logger.info(f"Elaborazione con UUID {elaboration_uuid} aggiornata.")
            else:
                # Se non esiste, crea una nuova elaborazione
                new_elaboration = Elaboration.Elaboration(
                    uuid=elaboration_uuid,
                    name=name,
                    format=format,
                    size=size,
                    thumbnail=thumbnail,
                    video_uuid=video_uuid,
                    status=str(status.value)
                )
                session.add(new_elaboration)
                logger.info(f"Nuova elaborazione con UUID {elaboration_uuid} creata.")

            # Commit della transazione
            await session.commit()

        except Exception as e:
            # Rollback in caso di errori
            await session.rollback()
            logger.error(f"Errore durante il salvataggio dell'elaborazione: {e}")
            raise