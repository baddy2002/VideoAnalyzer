import logging
from sqlite3 import DatabaseError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from sqlalchemy.exc import IntegrityError, OperationalError, DatabaseError, ProgrammingError

from sqlalchemy.future import select

from app.CNN.frameConfrontator.posesConfrontation import frame_confrontation
from app.config.database import get_session
from app.config.settings import Settings
from app.managements import prefixs
from app.managements.mediapipe import SIMILARITY_THRESHOLD
from app.models.entities import FrameAngle, Video

settings = Settings()
logger = logging.getLogger(__name__)


# Lista per tracciare i websocket connessi
connected_websockets = {}

video_analyze_router = APIRouter(
    prefix=prefixs.analyze_prefix,
    tags=["Keypoints analysis"],
    responses={404: {"description": "URL Not found"}},
)

@video_analyze_router.websocket("/upload_video")
async def video_processing(websocket: WebSocket, connection_uid: str):
    logger.info("WebSocket connection attempt")
    await websocket.accept()

    # Salva il WebSocket nella lista
    connected_websockets[connection_uid] = websocket

    while True:
        try:
            socketData = await websocket.receive_text()

            await websocket.send_json({"message": "alive"})

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            break  # Esci dal ciclo quando il client si disconnette

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            await websocket.send_json({"error": "An error occurred while processing the frame of your video"+str(e)})
            break  # Esci dal ciclo in caso di errore critico
