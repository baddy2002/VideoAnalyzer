

# Funzione per disegnare lo scheletro
import cv2
import logging

from managements.mediapipe import mp_pose
from app.CNN.frameAnalyzer.poseExtrapolation import find_keypoint_by_id

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
