import cv2
import numpy as np
import mediapipe as mp
import logging
import io
import tempfile
from app.CNN.frameConfrontator.poseExtrapolation import extract_keypoints, filter_keypoints
from app.CNN.frameConfrontator.posesConfrontation import calculate_pose_similarity_with_angles
from app.CNN.frameConfrontator.posesDraw import compare_skeleton, draw_skeleton
from utils.FileUtils import FileUtils
import os
from config.settings import Settings
from managements.mediapipe import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)
settings = Settings()


#<=================================================ANALYSIS================================================>

async def single_frame_confrontation(file1, file2, extension1, extension2, area, portions):
    file_content1 = file1.getvalue()
    file_content2 = file2.getvalue()

    # Verifica se i contenuti non sono vuoti
    if not file_content1 or not file_content2:
        raise ValueError("I file sono vuoti o non sono stati letti correttamente.")
    
    # Decodifica le immagini
    image1 = cv2.imdecode(np.frombuffer(file_content1, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(file_content2, np.uint8), cv2.IMREAD_COLOR)

    similar = None

    angle_keypoints = filter_keypoints(area, portions)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image1.flags.writeable = False           
    image2.flags.writeable = False              #reference: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md
    # Elenco dei keypoints
    keypoints1 = extract_keypoints(image1, angle_keypoints)
    keypoints2 = extract_keypoints(image2, angle_keypoints)

    
    # Calcola la similarità delle pose
    similarity_score, angles_results = calculate_pose_similarity_with_angles(keypoints1, keypoints2, angle_keypoints, area, portions)

    # Stampa il punteggio di similarità
    logger.info(f'Punteggio di similarità tra le pose: {similarity_score}')
    logger.info(f'Risultato comparazione angoli: {angles_results}')


    image1.flags.writeable = True           
    image2.flags.writeable = True
    if similarity_score != float("inf"):

        # Imposta una nuova soglia, ad esempio 5.0
        if similarity_score < SIMILARITY_THRESHOLD:
            logger.info("Le due pose sono simili.")
            similar = True
        else:
            similar = False
            logger.info("Le due pose sono diverse.")


        # Disegna gli scheletri sulle immagini
        draw_skeleton(image1, keypoints1, (0, 255,0))
        draw_skeleton(image2, keypoints1, (255, 0,0))
        compare_skeleton(image2, keypoints1, keypoints2, angles_results, angle_keypoints)
    else:
        draw_skeleton(image2, keypoints1, (255, 0,0))
        logger.warning("total confidence is insufficient, impossible to compare")
    # Codifica le immagini in memoria
    success1, encoded_image1 = cv2.imencode(extension1, image1)
    success2, encoded_image2 = cv2.imencode(extension2, image2)

    if not success1 or not success2:
        raise ValueError("Errore durante la codifica delle immagini.")

    # Crea byte stream dalle immagini codificate
    image_stream1 = io.BytesIO(encoded_image1)
    image_stream2 = io.BytesIO(encoded_image2)

    return similar, image_stream1, image_stream2
    



async def analyze_video_frames(file1, file2, extension1, extension2, area, portions):

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension1) as temp_video1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=extension2) as temp_video2:

         # Scrive i contenuti dei file video nei file temporanei
        temp_video1.write(await file1.read())
        temp_video2.write(await file2.read())

        # Chiude i file temporanei per essere utilizzabili con OpenCV
        temp_video1.close()
        temp_video2.close()

        # Aprire i video con OpenCV utilizzando i percorsi dei file temporanei
        video1 = cv2.VideoCapture(temp_video1.name)
        video2 = cv2.VideoCapture(temp_video2.name)

        # Verifica che i video siano stati aperti correttamente
        if not video1.isOpened() or not video2.isOpened():
            raise ValueError("Uno o entrambi i video non possono essere aperti.")

        # Ottieni informazioni sul video 2 (dimensioni, FPS, etc.)
        width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video2.get(cv2.CAP_PROP_FPS)

        # Determina l'estensione corretta dal file video di input
        output_filename = f'{settings.VIDEO_FOLDER}/output{extension2}'
        fourcc = cv2.VideoWriter_fourcc(*FileUtils.get_fourcc_from_extension(extension2))

        # Crea un video writer per salvare il video modificato
        output_video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        similar_frames = []

        # Definiamo l'estensione per i singoli frame (ad esempio .jpg)
        frame_extension = '.jpg'

        while True:
            # Leggi un frame da ciascun video
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()

            # Se uno dei video ha finito i frame, interrompi il ciclo
            if not ret1 or not ret2:
                break

            # Converte i frame in stream (necessario per single_frame_confrontation)
            frame_stream1 = io.BytesIO(cv2.imencode(frame_extension, frame1)[1].tobytes())
            frame_stream2 = io.BytesIO(cv2.imencode(frame_extension, frame2)[1].tobytes())

            # Chiamare la funzione esistente per confrontare i frame
            similar, _, processed_frame2 = await single_frame_confrontation(
                frame_stream1, frame_stream2, frame_extension, frame_extension, area, portions
            )

            # Decodifica i frame elaborati e scrivi nel video di output
            decoded_frame2 = cv2.imdecode(np.frombuffer(processed_frame2.getbuffer(), np.uint8), cv2.IMREAD_COLOR)
            output_video.write(decoded_frame2)

            # Aggiungi il risultato alla lista dei frame simili
            similar_frames.append(similar)


        # Rilascia i video e chiudi il video writer
        video1.release()
        video2.release()
        output_video.release()

        # Chiude i buffer dei file
        file1.file.close()
        file2.file.close()

    # Rimuovere i file temporanei
    os.remove(temp_video1.name)
    os.remove(temp_video2.name)
    return similar_frames, output_filename



#TODO: parallel processing of the frames