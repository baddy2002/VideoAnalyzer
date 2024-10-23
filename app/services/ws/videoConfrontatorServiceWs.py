import logging
import os
from sqlite3 import DatabaseError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from sqlalchemy.exc import IntegrityError, OperationalError, DatabaseError, ProgrammingError
import shutil
from sqlalchemy.future import select

from app.CNN.frameConfrontator.posesConfrontation import frame_confrontation
from app.config.database import get_session
from app.config.settings import Settings
from app.managements import prefixs
from app.managements.mediapipe import num_frames_to_check
from app.models.entities import ElaborationFrames, FrameAngle, Video
from uuid import UUID

from app.services.TransactionalDbService import save_elaboration
from app.services.movesDesignerService import create_elaboration_video, create_frame_image
settings = Settings()
logger = logging.getLogger(__name__)

video_confront_router = APIRouter(
    prefix=prefixs.confrontation_prefix,
    tags=["Confront videos in realtime"],
    responses={404: {"description": "URL Not found"}},
)

@video_confront_router.websocket("/keypoints_stream")
async def video_stream(websocket: WebSocket, elaboration_uuid: str):
    logger.info("WebSocket connection attempt")
    await websocket.accept()
    
    while True:
        try:
            # Riceve i dati JSON dal frontend
            frame_data = await websocket.receive_text()
            data = json.loads(frame_data)  # Deserializza i dati JSON
            logger.info("Frame data received: %s", data)
            try:
                uuid = data.get('video_uuid')
                if not uuid:
                    await websocket.send_json({"error": "Missing UUID in the request"})
                    continue  # Aspetta il prossimo messaggio
                async with get_session() as session:
                # Costruisci la query per cercare un singolo video
                    query = select(Video.Video).where(Video.Video.uuid == uuid)
                    
                    # Esegui la query
                    result = await session.execute(query)
                    video = result.scalars().first()
                    
                    # Se non viene trovato il video, ritorna un errore 404
                    if video is None:
                        await websocket.send_json({"error": "Impossible found a video with uuid: " + str(uuid)})
                        continue
            except DatabaseError as var1:
                logger.error(f"Database error: {var1}")
                await websocket.send_json({"error": "Database connection error, please try again later"+str(var1)})
                break
            except (IntegrityError, OperationalError, ProgrammingError) as var2:
                logger.error(f"SQL query error: {var2}")
                await websocket.send_json({"error": "Error executing SQL query"+str(var2)})
                continue
            except json.JSONDecodeError as var3:
                logger.error(f"Error decoding JSON: {var3}")
                await websocket.send_json({"error": "Invalid JSON format"+str(var3)})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred"+str(e)})
                break


            try:
                frame_number = data.get('frameNumber')
                if not frame_number:
                    await websocket.send_json({"error": "Missing frameNumber in the request"})
                    continue  # Aspetta il prossimo messaggio
            except json.JSONDecodeError as var3:
                logger.error(f"Error decoding JSON: {var3}")
                await websocket.send_json({"error": "Invalid JSON format"})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred"})
                break


            try:
                keypoints = data.get('landmarks')
                if not keypoints:
                    await websocket.send_json({"error": "Missing landmarks in the request"})
                    continue  # Aspetta il prossimo messaggio
            except json.JSONDecodeError as var3:
                logger.error(f"Error decoding JSON: {var3}")
                await websocket.send_json({"error": "Invalid JSON format"})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred"})
                break

            frame_range = range(frame_number - num_frames_to_check, frame_number + num_frames_to_check)
            try:
                async with get_session() as session:
                # Costruisci la query per cercare un singolo video
                    query = select(FrameAngle.FrameAngle).where(FrameAngle.FrameAngle.frame_number.in_(frame_range), FrameAngle.FrameAngle.video_uuid == uuid)
                    
                    # Esegui la query
                    result = await session.execute(query)
                    frame_analysis_list = result.scalars().all()
                    
                    if frame_analysis_list is None:
                        await websocket.send_json({"error": "Impossible found an analysis for frame: " + str(frame_number) + " of video: " + str(uuid)})
                        break
            except DatabaseError as var1:
                logger.error(f"Database error: {var1}")
                await websocket.send_json({"error": "Database connection error, please try again later"+str(var1)})
                break
            except (IntegrityError, OperationalError, ProgrammingError) as var2:
                logger.error(f"SQL query error: {var2}")
                await websocket.send_json({"error": "Error executing SQL query"+str(var2)})
                continue
            except json.JSONDecodeError as var3:
                logger.error(f"Error decoding JSON: {var3}")
                await websocket.send_json({"error": "Invalid JSON format"+str(var3)})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred"+ str(e)})
                break

            try:
                is_mirrored = data.get('is_mirrored')
                if is_mirrored is None:
                    await websocket.send_json({"error": "Missing is_mirrored param in the request"})
                    continue  # Aspetta il prossimo messaggio
            except json.JSONDecodeError as var3:
                logger.error(f"Error decoding JSON: {var3}")
                await websocket.send_json({"error": "Invalid JSON format"})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred"})
                break
            

            pose_connections = []
            i=0
            preference_frame_analysis = None
            for single_frame_analysis in frame_analysis_list:
                if single_frame_analysis.frame_number == frame_number:
                    preference_frame_analysis = single_frame_analysis
                # Ottieni le connessioni per il frame corrente
                current_pose_connections = frame_confrontation(keypoints, single_frame_analysis.angles_results, video.area, video.portions, frame_number, is_mirrored)

                for new_connection in current_pose_connections:
                    connection_key = new_connection['connection']
                    
                    # Controlla se la connessione è già in pose_connections
                    existing_conn = next((conn for conn in pose_connections if conn['connection'] == connection_key), None)
                    
                    if existing_conn:
                        # Aggiorna se angle_diff è minore
                        if new_connection['diff'] < existing_conn['diff']:
                            existing_conn.update({
                                'connection': existing_conn['connection'],
                                'color': new_connection['color'],
                                'frame_number': new_connection['frame_number'],
                                'diff': new_connection['diff']
                            })
                    # Se la connessione non esiste già, non fare nulla

                # Imposta le connessioni iniziali solo alla prima iterazione
                if i == 0:
                    pose_connections = current_pose_connections.copy()
                        
            logger.info('Risultato comparazione angoli:' +str(pose_connections))


            # Salva i dati nel database
            async with get_session() as session:
                new_frame_confrontation = ElaborationFrames.ElaborationFrames(
                    frame_number=frame_number,
                    keypoints=keypoints,
                    correct_keypoints=preference_frame_analysis.keypoints,  # Il keypoint corretto dal server
                    connections=pose_connections,  # Risultato comparazione angoli
                    elaboration_uuid=UUID(elaboration_uuid)  # UUID dell'elaborazione inviato dal client
                )
                session.add(new_frame_confrontation)
                await session.commit()
            temp_frames_dir = await create_frame_image(keypoints, frame_number, elaboration_uuid, video.height, video.width)
            # Invia una risposta se necessario
            await websocket.send_json(pose_connections)
            if preference_frame_analysis.is_last_frame:
                file_path, thumbnail_base64  = await create_elaboration_video(elaboration_uuid, fps=video.fps)
                await save_elaboration(
                    elaboration_uuid=elaboration_uuid,
                    name='elaboration_'+str(elaboration_uuid), 
                    format="mp4", 
                    size=os.path.getsize(os.path.join(settings.VIDEO_FOLDER, file_path)), 
                    thumbnail=thumbnail_base64,
                    video_uuid=uuid
                )
                for frame_file in os.listdir(temp_frames_dir):
                    os.remove(os.path.join(temp_frames_dir, frame_file))

                # Usa shutil.rmtree per rimuovere la directory temporanea
                shutil.rmtree(temp_frames_dir)

                break
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            break  # Esci dal ciclo quando il client si disconnette

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            await websocket.send_json({"error": "An error occurred while processing the frame"+str(e)})
            break  # Esci dal ciclo in caso di errore critico
