

# Funzione per disegnare lo scheletro
import cv2
import logging

from managements.mediapipe import mp_pose
from app.CNN.frameConfrontator.posesConfrontation import find_keypoint_by_id

logger = logging.getLogger(__name__)

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
            for angle_points, lato, portion in angle_keypoints:  
                if i in angle_points:
                    # Verifica se esiste un risultato di similarità per questo angolo
                    angle_result = angles_results.get(tuple(angle_points), None)
                    if angle_result:
                        total_counter += 1
                        if angle_result['similar']:
                            similar_count += 1
                        else:
                            logger.debug('angolo non simile: ' + str(angle_points) + ' risultato: ' + str(angle_result))
                    else:
                        logger.debug('angolo non trovato: ' + str(angle_points))
            
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