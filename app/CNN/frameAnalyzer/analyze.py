import cv2
import numpy as np
import mediapipe as mp
import logging
import io

logger = logging.getLogger(__name__)

# Inizializzazione di MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#TODO: Which should be? 
pose = mp_pose.Pose(static_image_mode=True,  model_complexity=2)   #more accurated but not appropriated for video
#pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)    #better and more optimized for video analysis

angle_keypoints_mapping = [
    ((11, 13, 15), 'sx', ['arms', 'body']),  # Braccio destro (spalla-gomito-polso)
    ((12, 14, 16), 'dx', ['arms', 'body']),  # Braccio sinistro (spalla-gomito-polso)
    ((23, 25, 27), 'sx', ['legs', 'body']), # Gamba destra (anca-ginocchio-caviglia)
    ((24, 26, 28), 'dx', ['legs', 'body']),  # Gamba sinistra (anca-ginocchio-caviglia)
    ((11, 23, 25), 'sx', ['body', 'legs']),  # Lato destro (spalla-anca-ginocchio)
    ((12, 24, 26), 'dx', ['body', 'legs']),  # Lato sinistro (spalla-anca-ginocchio)
    ((13, 15, 17), 'sx', ['arms']),  # Segmento gomito-polso destri
    ((14, 16, 18), 'dx', ['arms']),  #Segmento gomito-polso sinistri
    ((25, 27, 29), 'sx', ['legs']),  # Segmento ginocchio-caviglia
    ((26, 28, 30), 'dx', ['legs']),  # Segmento ginocchio-caviglia

    # Nuovi angoli per le dita delle mani
    ((15, 17, 19), 'dx', ['hands']),  # Dito indice destro (polso-pollice)
    ((15, 19, 21), 'dx', ['hands']),  # Dito medio destro (polso-indice)
    ((16, 18, 20), 'sx', ['hands']),  # Dito indice sinistro (polso-pollice)
    ((16, 20, 22), 'sx', ['hands']),  # Dito medio sinistro (polso-indice)

    # Nuovi angoli per le dita dei piedi
    ((27, 29, 31), 'dx', ['feets']),  # Dita piede destro (caviglia-dito grande)
    ((28, 30, 32), 'sx', ['feets']),  # Dita piede sinistro (caviglia-dito grande)
    # Angoli per la faccia
    ((0, 1, 3), None, ['head']), # Lato destro della faccia (orecchio-naso)
    ((0, 2, 3), None, ['head']),  # Lato sinistro della faccia (orecchio-naso)
    ((1, 0, 2), None, ['head']),  # Segmento che unisce orecchie attraverso la testa

    ((18, 20, 19), 'dx', ['hands']),  # Distanza tra il mignolo destro e l'indice destro
    ((18, 20, 19), 'sx', ['hands']),  # Distanza tra il mignolo sinistro e l'indice sinistro

    ((3, 7, 6), 'sx', ['head']),      # Angolo tra occhio sinistro e orecchio sinistro
    ((6, 8, 7), 'dx', ['head']),      # Angolo tra occhio destro e orecchio destro

    ((4, 5, 6), 'sx', ['arms']),      # Angolo tra spalla sinistra e gomito sinistro
    ((4, 6, 5), 'dx', ['arms']),      # Angolo tra spalla destra e gomito destro

    ((5, 6, 7), 'sx', ['arms']),      # Angolo tra gomito sinistro e polso sinistro
    ((6, 7, 8), 'dx', ['arms']),      # Angolo tra gomito destro e polso destro

    ((12, 24, 26), 'dx', ['legs']),   # Angolo tra spalla destra e ginocchio destro
    ((11, 23, 25), 'sx', ['legs']),   # Angolo tra spalla sinistra e ginocchio sinistro

    ((23, 24, 26), 'dx', ['legs']),   # Angolo tra ginocchio destro e caviglia destra
    ((22, 23, 25), 'sx', ['legs']),   # Angolo tra ginocchio sinistro e caviglia sinistra
    
    ((9, 10, 11),  None, ['head']),  # Angolo tra bocca sinistra e bocca destra
    ((0, 4, 5), None, ['head']),   # Angolo tra naso e spalle
    ((11, 12, 13), None, ['arms']), # Angolo tra gomito sinistro e gomito destro

    ((11, 12, 24), None, ['body','legs']),  # Angolo tra spalla destra, spalla sinistra e ginocchio destro
    ((24, 23, 12), None, ['body','legs']),  # Angolo tra ginocchio destro, anca destra e anca sinistra
    ((11, 24, 23), 'dx', ['hands']),  # Angolo tra spalla destra, ginocchio destro e anca destra
    ((12, 23, 24), 'sx', ['hands']),  # Angolo tra spalla sinistra, ginocchio sinistro e anca sinistra
    
]
    
]
# Imposta la tolleranza per gli angoli e la soglia
ANGLE_TOLERANCE = 10  # in gradi
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
        elif area is None:
            # Se è None, scegli l'area con il valore maggiore (dx o sx)
            if selected_area["dx"] > selected_area["sx"]:
                selected_area_label = "dx"
            elif selected_area["sx"] > selected_area["dx"]:
                selected_area_label = "sx"
            else:
                selected_area_label = "dx"  # Indifferente se sono uguali
        if selected_area_label is None:
            continue
        # Determina la porzione selezionata (portion) con il valore maggiore
        selected_portion = None
        max_portion_value = -1
        for portion in portions:
            if selected_portions[portion] > max_portion_value:
                max_portion_value = selected_portions[portion]
                selected_portion = portion

        # Aggiungi i keypoints, l'area e la porzione filtrata se l'area selezionata ha un valore > 0
        if max_portion_value > 0:
            filtered_keypoints.append((keypoints, selected_area_label, selected_portion))

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
            all_keypoints.append([idx, lm.x, lm.y, lm.visibility])  # Aggiungi l'indice per il mapping

        # Filtrare i keypoints che sono presenti in almeno un angolo
        angle_keypoints_unique = set()

        for mapping in angle_keypoints:
            angle_keypoints_unique.update(mapping[0])

        # Mantieni solo i keypoints filtrati
        keypoints = [all_keypoints[idx] for idx in angle_keypoints_unique if idx < len(all_keypoints)]
    
    return keypoints

# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)  # Vettore AB
    bc = np.array(c) - np.array(b)  # Vettore BC
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
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
            angles_results[(k1, k2, k3)] = {'angle1': angle1, 'angle2': angle2, 'similar': similar}
            
            if not similar:
                total_angle_difference += angle_diff * conf
            total_confidence += conf

    if total_confidence == 0:
        return float('inf'), angles_results
    
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
    file_content1 = await file1.read()
    file_content2 = await file2.read()

    # Verifica se i contenuti non sono vuoti
    if not file_content1 or not file_content2:
        raise ValueError("I file sono vuoti o non sono stati letti correttamente.")
    
    # Decodifica le immagini
    image1 = cv2.imdecode(np.frombuffer(file_content1, np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(file_content2, np.uint8), cv2.IMREAD_COLOR)

    similar = False

    angle_keypoints = filter_keypoints(area, portions)

    # Elenco dei keypoints
    keypoints1 = extract_keypoints(image1, angle_keypoints)
    keypoints2 = extract_keypoints(image2, angle_keypoints)

    
    # Calcola la similarità delle pose
    similarity_score, angles_results = calculate_pose_similarity_with_angles(keypoints1, keypoints2, angle_keypoints, area, portions)

    # Stampa il punteggio di similarità
    print(f'Punteggio di similarità tra le pose: {similarity_score}')
    print(f'Risultato comparazione angoli: {angles_results}')
    # Imposta una nuova soglia, ad esempio 5.0
    if similarity_score < SIMILARITY_THRESHOLD:
        print("Le due pose sono simili.")
        similar = True
    else:
        print("Le due pose sono diverse.")

    # Disegna gli scheletri sulle immagini
    draw_skeleton(image1, keypoints1, (0, 255,0))
    draw_skeleton(image2, keypoints1, (255, 0,0))
    compare_skeleton(image2, keypoints1, keypoints2, angles_results, angle_keypoints)

    # Codifica le immagini in memoria
    success1, encoded_image1 = cv2.imencode(extension1, image1)
    success2, encoded_image2 = cv2.imencode(extension2, image2)

    if not success1 or not success2:
        raise ValueError("Errore durante la codifica delle immagini.")

    # Crea byte stream dalle immagini codificate
    image_stream1 = io.BytesIO(encoded_image1)
    image_stream2 = io.BytesIO(encoded_image2)

    return similar, image_stream1, image_stream2
    


