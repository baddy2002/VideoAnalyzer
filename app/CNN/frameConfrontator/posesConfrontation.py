import math
import numpy as np
import logging

from managements.mediapipe import ANGLE_ERROR_TOLERANCE_THRESHOLD, ANGLE_WARNING_TOLERANCE_THRESHOLD, ANGLES_CONFIDENCE_TOLERANCE, ANGLE_TOLERANCE, ANGLE_CONFIDENCE_TOLERANCE, KEYPOINTS_CONFIDENCE_TOLERANCE


logger = logging.getLogger(__name__)


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


def calculate_pose_angles(keypoints1, angle_keypoints, selected_area, selected_portions):
    angles_results = []
    # Scorri attraverso ogni angolo nel dizionario `angle_keypoints`
    for key, value in angle_keypoints.items():
        # Estrai i keypoints k1, k2, k3 dai nomi delle chiavi, ad esempio "(11, 12, 24)"
        k1, k2, k3 = map(int, key.strip('()').split(', '))

        # Trova i keypoints per i rispettivi ID
        kp1_1 = keypoints1[k1] if k1 < len(keypoints1) else None
        kp1_2 = keypoints1[k2] if k2 < len(keypoints1) else None
        kp1_3 = keypoints1[k3] if k3 < len(keypoints1) else None

        # Se uno dei keypoints non è trovato, salta questa iterazione
        if kp1_1 is None or kp1_2 is None or kp1_3 is None:
            continue
        # Recupera la confidenza di ogni keypoint
        kp1_conf1, kp1_conf2, kp1_conf3 = kp1_1['visibility'], kp1_2['visibility'], kp1_3['visibility']

        area = value['area']
        portion = value['portion']
        # Calcola la confidenza complessiva per i keypoints attuali
        conf = (kp1_conf1 + kp1_conf2 + kp1_conf3) / 3 * selected_area.get(area, 0) * selected_portions.get(portion, 0)
        if conf < ANGLE_CONFIDENCE_TOLERANCE:
            continue
        # Calcola l'angolo con le coordinate [x, y] per i keypoints1
        angle1 = calculate_angle((kp1_1['x'], kp1_1['y']), 
                                    (kp1_2['x'], kp1_2['y']), 
                                    (kp1_3['x'], kp1_3['y']))
        # Prendi l'angolo e la confidenza da `angle_keypoints`
        angle_key = value['angle']


        # Calcola la differenza tra gli angoli
        angle_diff = abs(angle1 - angle_key) 
        angle_diff = angle_diff if angle_diff < 180 else 360 - angle_diff

        angles_results.append({
            'angle_keypoints': (k1, k2, k3),
            'angle_diff': angle_diff,
        })
    logger.info("calculated pose similarity: " + str(angles_results))
    return angles_results



def frame_confrontation(keypoints, angle_results, area, portions, frame_number):
    # Calcola la differenza degli angoli
    angle_differences = calculate_pose_angles(keypoints1=keypoints, angle_keypoints=angle_results, selected_area=area, selected_portions=portions)

    # Inizializza una lista per le connessioni delle pose
    pose_connections = []

    for angle_result in angle_differences:
        k1, k2, k3 = angle_result['angle_keypoints']
        angle_diff = angle_result['angle_diff']

        # Determina il colore basato sulla differenza angolare
        if angle_diff > ANGLE_ERROR_TOLERANCE_THRESHOLD:
            color = "#FF0000"
        elif angle_diff > ANGLE_WARNING_TOLERANCE_THRESHOLD:
            color = "#FFFF00"
        else:
            color = "#00FF00"

        # Aggiungi le connessioni alle pose
        new_color1 = update_connection((k1, k2), color, pose_connections)
        new_color2 = update_connection((k2, k3), color, pose_connections)
        pose_connections.append({"connection": (k1, k2), "color": new_color1, "frame_number": frame_number})
        pose_connections.append({"connection": (k2, k3), "color": new_color2, "frame_number": frame_number})
    
    return pose_connections

''' 
       if k2 == 14 and k3 == 16:
            logger.error("connection should be sended: " + str(k2, k3))
        
        elif k1 == 16 and (k2 == 18 or k2 == 20):
            logger.error("connection should be sended: " + str(k1, k2))
        elif (k2 == 18 or k2 == 20) and (k3 == 20 or k3 == 22):
                 logger.error("connection should be sended: " + str(k2, k3))
'''

        

# Funzione per trovare e aggiornare una connessione esistente
def update_connection(connection, new_color, pose_connections):
    average_color = None
    for conn, color, frame_number in pose_connections:
        if conn == connection:
            average_color = math.ceil((color_priority[new_color] + color_priority[color]) / 2)
            priority_color[average_color]
    return new_color


color_priority = {
    "#00FF00": 0,  # Green
    "#99FF00": 1,  # Green-Yellow (via di mezzo)
    "#FFFF00": 2,  # Yellow
    "#FFA500": 3,  # Orange
    "#FF0000": 4   # Red
}
priority_color = {v: k for k, v in color_priority.items()}