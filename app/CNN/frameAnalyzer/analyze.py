import base64
import uuid
import cv2
import numpy as np
import mediapipe as mp
import logging
import io, tempfile
from app.services.TransactionalDbService import save_pose_angles_to_db, save_video_metadata, update_video_metadata
from app.CNN.frameAnalyzer.poseExtrapolation import extract_keypoints, filter_keypoints, calculate_pose_angles
from app.CNN.frameAnalyzer.poseDraw import draw_skeleton
import os
from app.utils.FileUtils import FileUtils
from config.settings import Settings
from app.config.database import get_session
from app.models.entities.FrameAngle import FrameAngle
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)
settings = Settings()


#<=================================================ANALYSIS================================================>

async def single_frame_extimation(file1, area, portions, frame_number, video_uuid):
    file_content1 = file1.getvalue()

    # Verifica se i contenuti non sono vuoti
    if not file_content1:
        raise ValueError("I file sono vuoti o non sono stati letti correttamente.")
    
    # Decodifica le immagini
    image1 = cv2.imdecode(np.frombuffer(file_content1, np.uint8), cv2.IMREAD_COLOR)


    angle_keypoints = filter_keypoints(area, portions)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image1.flags.writeable = False           #reference: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md            
    # Elenco dei keypoints
    keypoints1 = extract_keypoints(image1, angle_keypoints)
    
    # Calcola la similarità delle pose
    angles_results = calculate_pose_angles(keypoints1, angle_keypoints, area, portions)
    await save_pose_angles_to_db(frame_number=frame_number, angles_results=angles_results,  video_uuid=video_uuid, keypoints=keypoints1)
    # Stampa il punteggio di similarità
    logger.info(f'Risultato angoli: {angles_results}')
    image1.flags.writeable = True
    draw_skeleton(image1, keypoints1, (0, 255,0))
    
    # Codifica le immagini in memoria
    encoded, encoded_image = cv2.imencode('.jpg', image1)

    if not encoded:
        raise ValueError("Errore durante la codifica delle immagini.")

    # Crea byte stream dalle immagini codificate
    image_stream = io.BytesIO(encoded_image)

    return angles_results, image_stream
    



async def analyze_video_frames(temp_video1, extension1,  area, portions, video_name, description):
    frame_number = 0

    # Chiude i file temporanei per essere utilizzabili con OpenCV
    temp_video1.close()

    # Aprire i video con OpenCV utilizzando i percorsi dei file temporanei
    video1 = cv2.VideoCapture(temp_video1.name)

    # Verifica che i video siano stati aperti correttamente
    if not video1.isOpened():
        raise ValueError("Video non aperto.")

    # Definiamo l'estensione per i singoli frame (ad esempio .jpg)
    frame_extension = '.jpg'

    video_id = uuid.uuid4()  # Genera un UUID per il video
    video_data = {
        'uuid': video_id,
        'name': video_name,
        'format': extension1,
        'size': 0,  # Dimensione iniziale
        'area':area,
        'portions':portions,
        'description': description
    }
    await save_video_metadata(video_data)

    output_filename = os.path.join(settings.VIDEO_FOLDER, f'{video_name.split(extension1)[0]}{datetime.now().strftime("%Y%m%d_%H%M%S")}{extension1}')
    fourcc = cv2.VideoWriter_fourcc(*FileUtils.get_fourcc_from_extension(extension1))
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video1.get(cv2.CAP_PROP_FPS)
    
    #output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    

    temp_frames_dir = os.path.join(settings.VIDEO_FOLDER, f'{video_name}_frames')
    os.makedirs(temp_frames_dir, exist_ok=True)

    while True:
        # Leggi un frame da ciascun video

        ret1, frame1 = video1.read()

        # Se uno dei video ha finito i frame, interrompi il ciclo
        if not ret1:
            break
        

        # Converte i frame in stream (necessario per single_frame_confrontation)
        frame_stream1 = io.BytesIO(cv2.imencode(frame_extension, frame1)[1].tobytes())

        # Chiamare la funzione esistente per confrontare i frame
        _, processed_frame = await single_frame_extimation(
            frame_stream1, area, portions, frame_number, video_id
        )
        frame_number += 1
        
        decoded_frame =  cv2.imdecode(np.frombuffer(processed_frame.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
        frame_filename = os.path.join(temp_frames_dir, f"frame_{frame_number:06d}.jpg")
        cv2.imwrite(frame_filename, decoded_frame)
        
        #output_video.write(decoded_frame)
        if frame_number == 100:
            _, buffer = cv2.imencode('.jpeg', decoded_frame)  # Salva come JPEG
            thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
    # Rilascia i video e chiudi il video writer
    video1.release()
    #output_video.release()
    # Rimuovere i file temporanei
     # Utilizza FFmpeg per unire i frame processati in un video
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Sovrascrivi il file di output
        '-framerate', str(fps),
        '-i', os.path.join(temp_frames_dir, 'frame_%06d.jpg'),  # Input dei frame
        '-c:v', 'libx264',  # Codifica in H.264
        '-pix_fmt', 'yuv420p',  # Compatibilità con i browser
        output_filename
    ]

    subprocess.run(ffmpeg_cmd, check=True)

    # Rimuovi i frame temporanei
    for frame_file in os.listdir(temp_frames_dir):
        os.remove(os.path.join(temp_frames_dir, frame_file))
    os.rmdir(temp_frames_dir)
    # Rimuovere i file temporanei
    if os.path.exists(temp_video1.name):
        os.remove(temp_video1.name)
    else:
        logger.error("file not found")

    # TODO: Carica il file su un servizio di storage esterno
    #await upload_to_storage_service(output_filename)  # Implementa la funzione per caricare il video

    # Salva i metadati nel database

    video_data = {
        'name': os.path.basename(output_filename),
        'format': extension1,
        'size': os.path.getsize(output_filename),  # Ottieni la dimensione del file
        'area':area,
        'portions':portions,
        'description': description,
        'thumbnail': thumbnail_base64
    }
    await update_video_metadata(video_id, video_data)

