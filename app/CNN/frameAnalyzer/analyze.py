import cv2
import numpy as np
import mediapipe as mp
import logging
import io
import tempfile
from utils.FileUtils import FileUtils
import os
from config.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()
# Inizializzazione di MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


pose = mp_pose.Pose(running_mode="VIDEO", min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)    #better and more optimized for video analysis

angle_keypoints_mapping = [

    #<================================BRACCIA===============================>

    ((11, 13, 15), 'sx', ['arms']),  # Braccio sinistro (spalla-gomito-polso)
    ((12, 14, 16), 'dx', ['arms']),  # Braccio destro (spalla-gomito-polso)

    #<==================================GAMBA=================================>

    ((23, 25, 27), 'sx', ['legs',]), # Gamba destra (anca-ginocchio-caviglia)
    ((24, 26, 28), 'dx', ['legs']),  # Gamba sinistra (anca-ginocchio-caviglia)

    #<======================================CORPO===========================>


    ((11, 12, 24), None, ['body']),  # spalla sx, spalla dx, anca dx
    ((12, 11, 23), None, ['body']),  # spalla dx, spalla sx, anca sx
    ((23, 24, 12), None, ['body']),  # anca sx, anca dx, spalla dx
    ((24, 23, 11), None, ['body']),  # anca sx, anca dx, spalla dx


    #<==========================MANI==========================>

    ((15, 17, 19), 'sx', ['hands']),  # polso-mignolo-indice
    ((15, 19, 21), 'sx', ['hands']),  # polso-indice-pollice
    ((16, 18, 20), 'dx', ['hands']),  # polso-mignolo-indice
    ((16, 20, 22), 'dx', ['hands']),  # polso-indice-pollice

    #<==========================PIEDI==========================>

    ((27, 29, 31), 'sx', ['feets']),  # caviglia-tallone-indice
    ((28, 30, 32), 'dx', ['feets']),  # caviglia-tallone-indice

    #<==========================TESTA==========================>
    ((0, 1, 2), "sx", ['head']), # naso-occhio interno-occhio 
    ((1, 2, 3), "sx", ['head']),  # occhio inerno- occhio -occhio esterno
    ((0, 4, 5), "sx", ['head']), # naso-occhio interno-occhio esterno
    ((4, 5, 6), "sx", ['head']),  # occhio inerno- occhio -occhio esterno
    ((1, 0, 4), None, ['head']),  # occhio sx- naso - occhio dx
    ((2, 3, 7), 'sx', ['head']),      # Angolo tra occhio sinistro e orecchio sinistro
    ((5, 6, 8), 'dx', ['head']),      # Angolo tra occhio destro e orecchio destro
    ((10, 9, 7),  "sx", ['head']),  # Angolo tra bocca sinistra e bocca destra e orecchio sx
    ((9, 10, 8),  "dx", ['head']),  # Angolo tra bocca sinistra e bocca destra e orecchio dx

    #<========================CORPO-BRACCIA======================================>
    ((23, 11, 13), 'sx', ['body', 'arms']),  # Lato sinistro (anca-spalla-gomito)
    ((24, 12, 14), 'dx', ['body', 'arms']),  # Lato sinistro (anca-spalla-gomito)
    ((12, 11, 13), None, ['body', 'arms']),  # spalla dx, spalla sx, gomito sx
    ((11, 12, 14), None, ['body', 'arms']),  # spalla dx, spalla sx, gomito sx

    #<========================CORPO-GAMBE======================================>
    ((11, 23, 25), 'sx', ['body', 'legs']),  # Lato sinistro (spalla-anca-ginocchio)
    ((12, 24, 26), 'dx', ['body', 'legs']),  # Lato destro (spalla-anca-ginocchio)
    ((23, 24, 26), None, ['body','legs']),   # anca sx - anca dx - ginocchio dx
    ((24, 23, 25), None, ['body', 'legs']),   # anca dx - anca sx - ginocchio sx
    
    #<========================MANI-BRACCIA======================================>
    ((13, 15, 17), 'sx', ['arms', 'hands']),  # Segmento gomito-polso-mignolo sinistri
    ((14, 16, 18), 'dx', ['arms', 'hands']),  #Segmento gomito-polso-mignolo destri
    ((13, 15, 21), 'sx', ['arms', 'hands']),  # Segmento gomito-polso-pollice sinistri
    ((14, 16, 22), 'dx', ['arms', 'hands']),  #Segmento gomito-polso-pollice destri

    #<========================PIEDI-GAMBE======================================>

    ((25, 27, 29), 'sx', ['legs', 'feets']),  # Segmento ginocchio-caviglia-tallone
    ((26, 28, 30), 'dx', ['legs', 'feets']),  # Segmento ginocchio-caviglia-tallone
    ((25, 27, 31), 'sx', ['legs', 'feets']),  # Segmento ginocchio-caviglia-indice
    ((26, 28, 32), 'dx', ['legs', 'feets']),  # Segmento ginocchio-caviglia-indice
    
]
    
# Imposta la tolleranza per gli angoli e la soglia
ANGLE_TOLERANCE = 10  # in gradi
KEYPOINTS_CONFIDENCE_TOLERANCE= 0.5
ANGLES_CONFIDENCE_TOLERANCE= 0.4
SIMILARITY_THRESHOLD = 7  

def filter_keypoints(selected_area, selected_portions):
    filtered_keypoints = []

    for mapping in angle_keypoints_mapping:
        keypoints, area, portions = mapping  # Estrai i valori dalla tupla
        selected_area_label = None
        # Determina l'area selezionata (sx, dx, None)
        if area == "dx" and selected_area["dx"] > 0:
            selected_area_label = "dx"
        elif area == "sx" and selected_area["sx"] > 0:
            selected_area_label = "sx"
        elif area is None and selected_area["dx"] > 0 and selected_area["sx"] > 0:   #se il keypoint ha area None l'utente deve specificare di volere un'analisi sia per parte dx che sx
            # Se è None, scegli l'area con il valore maggiore (dx o sx)
            if selected_area["dx"] > selected_area["sx"]:
                selected_area_label = "dx"
            elif selected_area["sx"] > selected_area["dx"]:
                selected_area_label = "sx"
            else:
                selected_area_label = "dx"  # Indifferente se sono uguali
        if selected_area_label is None:                                 #skippa i keypoints se non hanno area corretta
            continue
        # Determina la porzione selezionata (portion) con il valore maggiore
        selected_portion = None
        max_portion_value = -1
        for portion in portions:
            if selected_portions[portion] > 0:
                if selected_portions[portion] > max_portion_value:
                    max_portion_value = selected_portions[portion]
                    selected_portion = portion
        if selected_portion is None:
            continue
        # Aggiungi i keypoints, l'area e la porzione filtrata se l'area selezionata ha un valore > 0
        if max_portion_value > 0:
            filtered_keypoints.append((keypoints, selected_area_label, selected_portion))
    logger.info("filter keypoints: " + str(filtered_keypoints))
    return filtered_keypoints


# Funzione per estrarre i keypoints dalle immagini con MediaPipe
def extract_keypoints(image, angle_keypoints):
    # Converte l'immagine in RGB poiché MediaPipe lavora in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Processa l'immagine con MediaPipe
    results = pose.process(image_rgb)
    keypoints = []
    
    if results.pose_landmarks:
        # Estrarre tutti i keypoints dall'immagine
        all_keypoints = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > KEYPOINTS_CONFIDENCE_TOLERANCE:
                all_keypoints.append([idx, lm.x, lm.y, lm.visibility])  # Aggiungi l'indice per il mapping

        # Filtrare i keypoints che sono presenti in almeno un angolo
        angle_keypoints_unique = set()

        for mapping in angle_keypoints:
            angle_keypoints_unique.update(mapping[0])

        # Mantieni solo i keypoints filtrati
        keypoints = [all_keypoints[idx] for idx in angle_keypoints_unique if idx < len(all_keypoints)]
    logger.info("calculated keypoints: " + str(keypoints))
    return keypoints

# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)  # Vettore AB
    bc = np.array(c) - np.array(b)  # Vettore BC
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    logger.info("calculated angle " + str(angle))
    return np.degrees(angle)

# Funzione di supporto per trovare un keypoint per ID
def find_keypoint_by_id(keypoints, keypoint_id):
    for kp in keypoints:
        if kp[0] == keypoint_id:  # Il primo elemento è l'ID del keypoint
            return kp
    return None  # Se il keypoint non viene trovato

def calculate_pose_similarity_with_angles(keypoints1, keypoints2, angle_keypoints, selected_area, selected_portions):
    total_angle_difference = 0
    total_confidence = 0
    angles_results = {}

    for (k1, k2, k3), area, portion in angle_keypoints:
        # Cerca i keypoints per ID
        kp1_1 = find_keypoint_by_id(keypoints1, k1)
        kp1_2 = find_keypoint_by_id(keypoints1, k2)
        kp1_3 = find_keypoint_by_id(keypoints1, k3)
        
        kp2_1 = find_keypoint_by_id(keypoints2, k1)
        kp2_2 = find_keypoint_by_id(keypoints2, k2)
        kp2_3 = find_keypoint_by_id(keypoints2, k3)
        if kp1_1 is None or kp1_2  is None or kp1_3 is None or kp2_1 is None or kp2_2  is None or kp2_3 is None:
            continue
        #recupera confidenza di ogni kp
        kp1_conf1, kp1_conf2, kp1_conf3 = kp1_1[3], kp1_2[3], kp1_3[3]
        kp2_conf1, kp2_conf2, kp2_conf3 = kp2_1[3], kp2_2[3], kp2_3[3]
        
        # Controlla che tutti i keypoints siano presenti in entrambe le immagini
        if all(conf > 0.1 for conf in [kp1_conf1, kp1_conf2, kp1_conf3, kp2_conf1, kp2_conf2, kp2_conf3]):
            angle1 = calculate_angle(kp1_1[1:3], kp1_2[1:3], kp1_3[1:3])  # Coordinate [x, y] escludiamo 0 (id) e 3 (conf)
            angle2 = calculate_angle(kp2_1[1:3], kp2_2[1:3], kp2_3[1:3])
            angle_diff = abs(angle1 - angle2)
            angle_diff = min(angle_diff, 360 - angle_diff)  # Se maggiore di 180, prendi la differenza complementare
            # Se la differenza tra gli angoli è minore della tolleranza, considerali uguali
            print('differenza = ' + str(angle_diff))
            similar = angle_diff < ANGLE_TOLERANCE
            
            conf = min(((kp1_conf1 + kp1_conf2 + kp1_conf3) / 3), ((kp2_conf1 + kp2_conf2 + kp2_conf3) / 3))  # min della Media della confidenza dei punti degli angoli nell due figure
            conf = conf*selected_area[area]*selected_portions[portion]                #confidenza in base a quanto l'angolo ci interessa
                    # Salva il risultato per questo angolo
            angles_results[(k1, k2, k3)] = {
                'angle1': angle1,
                'angle2': angle2,
                'similar': similar,
                'angle_diff': angle_diff,
                'confidence': conf,
                'area': area,
                'portion': portion
            }
            
            if not similar:
                total_angle_difference += angle_diff * conf
            total_confidence += conf

    if total_confidence < ANGLES_CONFIDENCE_TOLERANCE:
        return float('inf'), angles_results
    logger.info("calculated pose similarity: " + str(angles_results))
    return total_angle_difference / total_confidence, angles_results


# Funzione per disegnare lo scheletro
def draw_skeleton(image, keypoints, color):
    # Disegna i keypoints se la confidenza è sufficiente
    for point in keypoints:
        if point[3] > 0.1:  # Solo se la confidenza è sufficiente
            cv2.circle(image, (int(point[1] * image.shape[1]), int(point[2] * image.shape[0])), 5, color, -1)

    # Definisci le connessioni
    connections = list(mp_pose.POSE_CONNECTIONS)  # Converti il frozenset in una lista

    for connection in connections:
        # Cerca i keypoints tramite il loro ID
        p1 = find_keypoint_by_id(keypoints, connection[0])
        p2 = find_keypoint_by_id(keypoints, connection[1])

        # Solo se entrambi i keypoints esistono e la loro confidenza è sufficiente
        if p1 is not None and p2 is not None and p1[3] > 0.1 and p2[3] > 0.1:
            cv2.line(image, (int(p1[1] * image.shape[1]), int(p1[2] * image.shape[0])),
                      (int(p2[1] * image.shape[1]), int(p2[2] * image.shape[0])), color, 2)

# Funzione per disegnare gli scheletri con evidenziazione dei collegamenti diversi
def compare_skeleton(image, keypoints1, keypoints2, angles_results, angle_keypoints):
    # Disegna i keypoints della seconda immagine in rosso o verde in base alla similarità
    for i, point in enumerate(keypoints2):
        if point[3] > 0.1:  # Solo se la confidenza è sufficiente
            similar_count = 0  # Numero di angoli simili
            total_counter = 0  # Numero totale di angoli coinvolti per il keypoint
            
            # Verifica tutti gli angoli che coinvolgono il keypoint i
            for angle_points, lato, portion in angle_keypoints:  # Corretto l'accesso
                if i in angle_points:
                    # Verifica se esiste un risultato di similarità per questo angolo
                    angle_result = angles_results.get(tuple(angle_points), None)
                    if angle_result:
                        total_counter += 1
                        if angle_result['similar']:
                            similar_count += 1
                        else:
                            print('angolo non simile: ' + str(angle_points) + ' risultato: ' + str(angle_result))
                    else:
                        print('angolo non trovato: ' + str(angle_points))
            
            # Se c'è almeno un angolo simile, disegna il keypoint in verde, altrimenti in rosso
            if similar_count:
                if similar_count == total_counter:
                    color = (0, 255, 0)  # Verde se simile
                else:
                    diff = (total_counter - similar_count) / total_counter     #percentuale angloli sbagliati
                    if diff > 0.5:
                        green = (255*(1-diff)+60)
                    else:
                        green = 255
                    color = (0, green, 255) # da giallo ad arancione più angoli sono sbagliati
            else:
                color = (0, 0, 255)  # Rosso se diverso

            # Disegna il keypoint
            cv2.circle(image, (int(point[1] * image.shape[1]), int(point[2] * image.shape[0])), 5, color, -1)
    connections = list(mp_pose.POSE_CONNECTIONS)

    # Disegna i collegamenti della seconda immagine in rosso o verde in base alla similarità
    for connection in connections:
        p1 = find_keypoint_by_id(keypoints2, connection[0])
        p2 = find_keypoint_by_id(keypoints2, connection[1])
        if p1 is None or p2 is None:
            continue
        if p1[3] > 0.1 and p2[3] > 0.1:  # Solo se la confidenza è sufficiente
            similar_found = False
            for angle_points, lato, portion in angle_keypoints:
                if connection[0] in angle_points and connection[1] in angle_points:
                    # Verifica se esiste un risultato di similarità per questo angolo
                    angle_result = angles_results.get(tuple(angle_points), None)
                    if angle_result and angle_result['similar']:
                        similar_found = True
                        break  # Esci se trovi un angolo simile
        
            if similar_found:
                color = (0, 255, 0)  # Verde se simile
            else:
                color = (0, 0, 255)  # Rosso se diverso
            
            cv2.line(image, (int(p1[1] * image.shape[1]), int(p1[2] * image.shape[0])),
                            (int(p2[1] * image.shape[1]), int(p2[2] * image.shape[0])), color, 2)

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

    # Elenco dei keypoints
    keypoints1 = extract_keypoints(image1, angle_keypoints)
    keypoints2 = extract_keypoints(image2, angle_keypoints)

    
    # Calcola la similarità delle pose
    similarity_score, angles_results = calculate_pose_similarity_with_angles(keypoints1, keypoints2, angle_keypoints, area, portions)

    # Stampa il punteggio di similarità
    print(f'Punteggio di similarità tra le pose: {similarity_score}')
    print(f'Risultato comparazione angoli: {angles_results}')

    if similarity_score != float("inf"):

        # Imposta una nuova soglia, ad esempio 5.0
        if similarity_score < SIMILARITY_THRESHOLD:
            print("Le due pose sono simili.")
            similar = True
        else:
            similar = False
            print("Le due pose sono diverse.")

        # Disegna gli scheletri sulle immagini
        draw_skeleton(image1, keypoints1, (0, 255,0))
        draw_skeleton(image2, keypoints1, (255, 0,0))
        compare_skeleton(image2, keypoints1, keypoints2, angles_results, angle_keypoints)
    else:
        logger.warn("total confidence is insufficient, impossible to compare")
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
            similar, processed_frame1, processed_frame2 = await single_frame_confrontation(
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