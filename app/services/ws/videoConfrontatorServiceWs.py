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
from app.managements.mediapipe import FIRST_FRAME_NUMBER, num_frames_to_check
from app.models.entities import ElaborationFrames, FrameAngle, Video
from uuid import UUID
from services import videoConfrontorService
from app.models.enums.ElaborationStatus import ElaborationStatus
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
                await websocket.send_json({"error": "Invalid JSON format: "+str(var3)})
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send_json({"error": "An unexpected error occurred: "+str(e)})
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
            

            await save_elaboration(
                elaboration_uuid=elaboration_uuid,
                name='elaboration_'+str(elaboration_uuid), 
                format="mp4", 
                size=0, 
                thumbnail='thumbnail_base64',
                video_uuid=uuid,
                status=ElaborationStatus.PROCESSING
            )


            pose_connections = []
            i=0
            total_min_x=1000
            min_x_key=0
            total_max_x=0
            min_y_key=0
            total_min_y=1000
            max_x_key=0
            total_max_y=0
            max_y_key=0
            preference_frame_analysis = None
            for single_frame_analysis in frame_analysis_list:
                #ricerca dei valori bounding box nei vari frame considerati
                if single_frame_analysis.min_x < total_min_x:
                    total_min_x = single_frame_analysis.min_x
                    min_x_key = single_frame_analysis.min_x_key
                if single_frame_analysis.min_y < total_min_y:
                    total_min_y = single_frame_analysis.min_y
                    min_y_key = single_frame_analysis.min_y_key
                if single_frame_analysis.max_x > total_max_x:
                    total_max_x = single_frame_analysis.max_x
                    max_x_key = single_frame_analysis.max_x_key
                if single_frame_analysis.max_y > total_max_y:
                    total_max_y = single_frame_analysis.max_y
                    max_y_key = single_frame_analysis.max_y_key
                
                if single_frame_analysis.frame_number == frame_number:
                    preference_frame_analysis = single_frame_analysis
                
                # Ottieni le connessioni per il frame corrente
                current_pose_connections, barycenter_x, barycenter_y = frame_confrontation(keypoints, single_frame_analysis.angles_results, video.area, video.portions, frame_number, is_mirrored)

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
                    else:
                        pose_connections = current_pose_connections.copy()
                    # Se la connessione non esiste già, non fare nulla

                # Imposta le connessioni iniziali solo alla prima iterazione
                if i == 0:
                    pose_connections = current_pose_connections.copy()
                i+=1
            
            delta_barycenter_x = abs(barycenter_x-preference_frame_analysis.barycenter_x) 
            delta_barycenter_y = abs(barycenter_y-preference_frame_analysis.barycenter_y)

            logger.info('Risultato comparazione angoli:' +str(pose_connections))
            in_box, all_green =videoConfrontorService.check_connection(pose_connections, 
                                                                       keypoints, 
                                                                       total_min_x, 
                                                                       min_x_key,
                                                                       total_max_x,
                                                                       max_x_key,
                                                                       total_min_y, 
                                                                       min_y_key,
                                                                       total_max_y,
                                                                       max_y_key,
                                                                       delta_barycenter_x=delta_barycenter_x,
                                                                       delta_barycenter_y = delta_barycenter_y,
                                                                       eps=0.1)
            if preference_frame_analysis:
                # Salva i dati nel database
                
                if in_box and (frame_number != FIRST_FRAME_NUMBER or all_green): #se è nel bounding box ed è tutto verde parte,
                                                                                   #oppure  se è già partito 
                    logger.info("all connections are green..." + str(all_green) + " in box: " + str(in_box))
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
                    temp_frames_dir = await create_frame_image(keypoints, frame_number, elaboration_uuid, video.height, video.width, pose_connections)
                # Invia una risposta se necessario
                
                if pose_connections and pose_connections[0]:         
                    pose_connections[0]['all_green'] = all_green
                    pose_connections[0]['in_box'] = in_box
                await websocket.send_json(pose_connections)
                if preference_frame_analysis.is_last_frame:
                    file_path, thumbnail_base64  = await create_elaboration_video(elaboration_uuid, fps=video.fps)
                    await save_elaboration(
                        elaboration_uuid=elaboration_uuid,
                        name='elaboration_'+str(elaboration_uuid), 
                        format="mp4", 
                        size=os.path.getsize(os.path.join(settings.VIDEO_FOLDER, file_path)), 
                        thumbnail=thumbnail_base64,
                        video_uuid=uuid,
                        status=ElaborationStatus.SAVED
                    )
                    for frame_file in os.listdir(temp_frames_dir):
                        os.remove(os.path.join(temp_frames_dir, frame_file))

                    # Usa shutil.rmtree per rimuovere la directory temporanea
                    shutil.rmtree(temp_frames_dir)
                    await websocket.send_json({"message": "completed"})
                    break
            
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
            break  # Esci dal ciclo quando il client si disconnette

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            await websocket.send_json({"error": "An error occurred while processing the frame"+str(e)})
            break  # Esci dal ciclo in caso di errore critico
