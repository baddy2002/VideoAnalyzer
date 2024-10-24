import base64
import cv2
import numpy as np
import os
import logging
from managements.mediapipe import mp_pose

from app.config.settings import Settings
import subprocess

logger = logging.getLogger(__name__)
settings = Settings()
connections = list(mp_pose.POSE_CONNECTIONS)

async def create_frame_image(keypoints, frame_number, elaboration_uuid, height, width):
    # Creiamo il path della cartella temporanea per salvare i frame
    temp_frames_dir = os.path.join(settings.VIDEO_FOLDER, f'{elaboration_uuid}_frames')
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # Creiamo un'immagine nera (sfondo nero)
    frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)


    # Disegnare le connessioni tra i keypoints
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            # Recupera le coordinate dei keypoints con il loro indice
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # Controlla la visibilità dei keypoints (0.0 è invisibile, 1.0 è completamente visibile)
            if start_point['visibility'] > 0.3 and end_point['visibility'] > 0.3:
                start_x = int(start_point['x'] * width)
                start_y = int(start_point['y'] * height)
                end_x = int(end_point['x'] * width)
                end_y = int(end_point['y'] * height)

                # Disegna una linea rossa tra i keypoints
                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 5)

    # Disegnare i keypoints sull'immagine
    for kp in keypoints:
        if kp['visibility'] > 0.1:  # Disegna solo keypoints con visibilità > 0.5
            x = int(kp['x'] * width)
            y = int(kp['y'] * height)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Disegna un punto rosso
    
    # Salva l'immagine come PNG con il nome basato sul numero di frame
    frame_filename = os.path.join(temp_frames_dir, f"frame_{int(frame_number/5):06d}.png")
    cv2.imwrite(frame_filename, frame)

    return temp_frames_dir

async def create_elaboration_video(elaboration_uuid, fps=30, thumbnail=None):
    # Path alla cartella dei frame
    temp_frames_dir = os.path.join(settings.VIDEO_FOLDER, f'{elaboration_uuid}_frames')
    
    # Trova il primo frame disponibile
    try:
        first_frame_path = os.path.join(temp_frames_dir, 'frame_000001.png')  # Modifica per usare il primo frame
        if not os.path.exists(first_frame_path):
            raise FileNotFoundError(f"First frame not found at {first_frame_path}")
        
        # Carica il primo frame come immagine
        first_frame = cv2.imread(first_frame_path)
        
        # Genera la thumbnail a partire dal primo frame
        _, buffer = cv2.imencode('.jpeg', first_frame)
        thumbnail_base64 = base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error("Impossible to generate thumbnail for elaboration with uuid " + str(elaboration_uuid) + ": " + str(e))
        thumbnail_base64 = 'dummy_thumbnail'


    # Nome del file video finale
    output_filename = os.path.join(settings.VIDEO_FOLDER, f'elaboration_{elaboration_uuid}.mp4')
    # Comando FFmpeg per unire i frame in un video
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Sovrascrivi il file di output
        '-framerate', str((fps/5)),  # Imposta il frame rate
        '-i', os.path.join(temp_frames_dir, 'frame_%06d.png'),  # Input dei frame in PNG
        '-c:v', 'libx264',  # Codifica in H.264
        '-pix_fmt', 'yuv420p',  # Compatibilità con i browser
        output_filename
    ]
    
    # Esegui il comando FFmpeg
    subprocess.run(ffmpeg_cmd, check=True)

    return output_filename, thumbnail_base64