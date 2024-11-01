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

async def create_frame_image(keypoints, frame_number, elaboration_uuid, height, width, pose_connections=None):
    # Creiamo il path della cartella temporanea per salvare i frame
    temp_frames_dir = os.path.join(settings.VIDEO_FOLDER, f'{elaboration_uuid}_frames')
    os.makedirs(temp_frames_dir, exist_ok=True)
    
    # Creiamo un'immagine nera (sfondo nero)
    frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)

    
    # Disegnare le connessioni tra i keypoints
    for connection in connections:
        color=(0, 0,255)
        start_idx, end_idx = connection
        if pose_connections:
            for item in pose_connections:
                if item.get("connection") == connection:
                    color = hex_to_bgr(item.get('color'))
                    break
        start_point = next((kp for kp in keypoints if kp['index'] == start_idx), None)
        end_point = next((kp for kp in keypoints if kp['index'] == end_idx), None)
        if start_point  and end_point:
            # Recupera le coordinate dei keypoints con il loro indice

            # Controlla la visibilità dei keypoints (0.0 è invisibile, 1.0 è completamente visibile)
            if start_point['visibility'] > 0.3 and end_point['visibility'] > 0.3:
                start_x = int(start_point['x'] * width)
                start_y = int(start_point['y'] * height)
                end_x = int(end_point['x'] * width)
                end_y = int(end_point['y'] * height)
                # Disegna una linea tra i keypoints
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 5)

    # Disegnare i keypoints sull'immagine
    for kp in keypoints:
        if kp['visibility'] > 0.3:  # Disegna solo keypoints con visibilità > 0.3
            x = int(kp['x'] * width)
            y = int(kp['y'] * height)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Disegna un punto rosso
    
    # Salva l'immagine come PNG con il nome basato sul numero di frame
    frame_filename = os.path.join(temp_frames_dir, f"frame_{int(frame_number/5):06d}.png")
    cv2.imwrite(frame_filename, frame)

    return temp_frames_dir

async def create_elaboration_video(elaboration_uuid, fps=30, thumbnail=None):
    # Path alla cartella dei frame
    temp_frames_dir = os.path.join(settings.VIDEO_FOLDER, f'{elaboration_uuid}_frames')

    # Trova il primo frame disponibile
    try:
        # Elenca i file nella cartella e filtra solo quelli con estensione .png
        frame_files = [f for f in os.listdir(temp_frames_dir) if f.endswith('.png')]
        
        # Se non ci sono file, solleva un'eccezione
        if not frame_files:
            raise FileNotFoundError(f"No frame files found in {temp_frames_dir}")

        # Ordina i file numericamente per estrarre il primo disponibile
        frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Estrae il numero dal nome del file
        first_frame_path = os.path.join(temp_frames_dir, frame_files[0])  # Prende il primo file trovato

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

    concat_file_path = os.path.join(temp_frames_dir, 'frames.txt')
    with open(concat_file_path, 'w') as f:
        for frame_file in frame_files:
            f.write(f"file '{os.path.join(temp_frames_dir, frame_file)}'\n")

    # Comando FFmpeg per unire i frame in un video utilizzando il file di testo
    ffmpeg_cmd = [
        'ffmpeg', '-y',  # Sovrascrivi il file di output
        '-r', str(fps/5),  # Imposta il frame rate
        '-f', 'concat',  # Specifica il formato di input come concat
        '-safe', '0',  # Consente l'uso di percorsi assoluti
        '-i', concat_file_path,  # Input del file di testo con l'elenco dei frame
        '-c:v', 'libx264',  # Codifica in H.264
        '-pix_fmt', 'yuv420p',  # Compatibilità con i browser
        output_filename
    ]

    try:
        # Esegui il comando FFmpeg
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing frames with FFmpeg: {str(e)}")
        raise

    return output_filename, thumbnail_base64


def hex_to_bgr(hex_color):
    # Rimuovi il simbolo "#" se presente
    hex_color = hex_color.lstrip("#")
    # Converti da RGB a BGR
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr