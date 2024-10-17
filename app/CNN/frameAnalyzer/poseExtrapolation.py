import logging
import cv2
import mediapipe as mp
import numpy as np

from managements.mediapipe import angle_keypoints_mapping, KEYPOINTS_CONFIDENCE_TOLERANCE, ANGLE_CONFIDENCE_TOLERANCE, pose

logger = logging.getLogger(__name__)

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
    
    # Opzionale: Visualizza i risultati sul frame
    if results.pose_landmarks:
        all_keypoints = []
        # Itera sui landmarks di pose
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if lm.visibility > KEYPOINTS_CONFIDENCE_TOLERANCE:
                all_keypoints.append([idx, lm.x, lm.y, lm.visibility])  
        
        # Filtrare i keypoints che sono presenti in almeno un angolo
        angle_keypoints_unique = set()
        logger.info("Angle keypoints: " + str(angle_keypoints))
        for mapping in angle_keypoints:
            angle_keypoints_unique.update(mapping[0])

        logger.info("Angle keypoints unique: " + str(angle_keypoints_unique))
        logger.info("all_keypoints value: " + str(all_keypoints))
        
        # Mantieni solo i keypoints filtrati
        keypoints = [all_keypoints[idx] for idx in angle_keypoints_unique if idx < len(all_keypoints)]

    logger.info("calculated keypoints: " + str(keypoints))
    return keypoints


# Funzione di supporto per trovare un keypoint per ID
def find_keypoint_by_id(keypoints, keypoint_id):
    for kp in keypoints:
        if kp[0] == keypoint_id:  # Il primo elemento è l'ID del keypoint
            return kp
    return None  # Se il keypoint non viene trovato


# Funzione per calcolare l'angolo tra tre punti
def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)  # Vettore AB
    bc = np.array(c) - np.array(b)  # Vettore BC
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    logger.info("calculated angle " + str(angle))
    return np.degrees(angle)

def calculate_pose_angles(keypoints1, angle_keypoints, selected_area, selected_portions):
    angles_results = {}

    for (k1, k2, k3), area, portion in angle_keypoints:
        # Cerca i keypoints per ID
        kp1_1 = find_keypoint_by_id(keypoints1, k1)
        kp1_2 = find_keypoint_by_id(keypoints1, k2)
        kp1_3 = find_keypoint_by_id(keypoints1, k3)

        if kp1_1 is None or kp1_2  is None or kp1_3 is None:
            continue
        #recupera confidenza di ogni kp
        kp1_conf1, kp1_conf2, kp1_conf3 = kp1_1[3], kp1_2[3], kp1_3[3]
        
        # Controlla che tutti i keypoints siano presenti in entrambe le immagini
        if all(conf > 0.1 for conf in [kp1_conf1, kp1_conf2, kp1_conf3]):
            angle1 = calculate_angle(kp1_1[1:3], kp1_2[1:3], kp1_3[1:3])  # Coordinate [x, y] escludiamo 0 (id) e 3 (conf)

            
            conf = ((kp1_conf1 + kp1_conf2 + kp1_conf3) / 3)
            conf = conf*selected_area[area]*selected_portions[portion]                #confidenza in base a quanto l'angolo ci interessa
            
            # Se la confidenza ottenuta dalla visione dell'angolo nel video e dall'interesse per esso non è sufficiente non lo considero
            if conf < ANGLE_CONFIDENCE_TOLERANCE:
                continue
            # Salva il risultato per questo angolo
            angles_results[(k1, k2, k3)] = {
                'angle': angle1,
                'confidence': conf,
                'area': area,
                'portion': portion
            }
    logger.info("calculated pose similarity: " + str(angles_results))
    return angles_results