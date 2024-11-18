import math
import numpy as np
import logging

from managements.mediapipe import angle_mirroring, ANGLE_ERROR_TOLERANCE_THRESHOLD, ANGLE_WARNING_TOLERANCE_THRESHOLD, ANGLE_CONFIDENCE_TOLERANCE 


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
    barycenter_x = 0
    barycenter_y = 0
    kp_checked = []
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
        
        if kp1_1['index'] not in kp_checked:
            barycenter_x += kp1_1['x']
            barycenter_y += kp1_1['y']
            kp_checked.append(kp1_1['index'])
        if kp1_2['index'] not in kp_checked:
            barycenter_x += kp1_2['x']
            barycenter_y += kp1_2['y']
            kp_checked.append(kp1_2['index'])   
        if kp1_3['index'] not in kp_checked:
            barycenter_x += kp1_3['x']
            barycenter_y += kp1_3['y']
            kp_checked.append(kp1_3['index']) 


        # Calcola la differenza tra gli angoli
        angle_diff = abs(angle1 - angle_key) 
        angle_diff = angle_diff if angle_diff < 180 else 360 - angle_diff

        angles_results.append({
            'angle_keypoints': (k1, k2, k3),
            'angle_diff': angle_diff,
        })
    logger.info("calculated pose similarity: " + str(angles_results))
    if len(kp_checked) > 0:
        barycenter_x = barycenter_x/len(kp_checked)
        barycenter_y = barycenter_y/len(kp_checked)
    else :
        barycenter_x=0
        barycenter_y=0
    return angles_results, barycenter_x, barycenter_y



def frame_confrontation(keypoints, angle_results, area, portions, frame_number, is_mirrored):
    # Calcola la differenza degli angoli
    if is_mirrored:
        angle_results = mirror_angles(angle_results, angle_mirroring)
    angle_differences, barycenter_x, barycenter_y = calculate_pose_angles(keypoints1=keypoints, angle_keypoints=angle_results, selected_area=area, selected_portions=portions)

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
        update_or_add_connection((k1, k2), new_color1, frame_number, angle_diff, pose_connections)
        update_or_add_connection((k2, k3), new_color2, frame_number, angle_diff, pose_connections)

    return pose_connections, barycenter_x, barycenter_y

def update_or_add_connection(connection, color, frame_number, angle_diff, pose_connections):
    # Cerca se la connessione esiste già
    for item in pose_connections:
        if item["connection"] == connection:
            # Se esiste, aggiorna i valori
            item["color"] = color
            item["frame_number"] = frame_number
            item["diff"] = angle_diff
            return

    # Se la connessione non esiste, aggiungi un nuovo elemento
    pose_connections.append({
        "connection": connection,
        "color": color,
        "frame_number": frame_number,
        "diff": angle_diff
    })

# Funzione per effettuare il mirroring degli angoli
def mirror_angles(angle_results, angle_mirroring):
    mirrored_results = {}

    for angle_key, angle_data in angle_results.items():
        # Converti la stringa chiave in una tupla
        angle_key_tuple = eval(angle_key)

        # Cerca se l'angolo corrente è presente nelle coppie di mirroring
        mirrored_angle_key = None
        for original, mirrored in angle_mirroring:
            if angle_key_tuple == original:
                mirrored_angle_key = mirrored
                break           #torna al ciclo principale
            elif angle_key_tuple == mirrored:
                mirrored_angle_key = original
                break

        # Se esiste una corrispondenza, sostituisci la chiave con l'angolo mirroring
        if mirrored_angle_key:
            mirrored_key_str = str(mirrored_angle_key)
            mirrored_results[mirrored_key_str] = angle_data
        else:
            # Se non esiste una corrispondenza, mantieni l'angolo originale
            mirrored_results[angle_key] = angle_data
    logger.info("mirrored: " + str(mirrored_results))
    return mirrored_results

        

# Funzione per trovare e aggiornare una connessione esistente
def update_connection(connection, new_color, pose_connections):
    average_color = None
    for conn, color, frame_number, _ in pose_connections:
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