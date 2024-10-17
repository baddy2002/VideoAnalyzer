import numpy as np
import logging

from managements.mediapipe import ANGLES_CONFIDENCE_TOLERANCE, ANGLE_TOLERANCE, ANGLE_CONFIDENCE_TOLERANCE


logger = logging.getLogger(__name__)


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
            
            # Se la confidenza ottenuta dalla visione dell'angolo nel video e dall'interesse per esso non è sufficiente non lo considero
            if conf < ANGLE_CONFIDENCE_TOLERANCE:
                continue
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