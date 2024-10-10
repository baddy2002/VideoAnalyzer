import logging
import cv2

from managements.mediapipe import angle_keypoints_mapping, KEYPOINTS_CONFIDENCE_TOLERANCE, pose

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