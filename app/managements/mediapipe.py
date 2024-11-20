from config.settings import Settings
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

settings = Settings()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Inizializzazione di MediaPipe Pose
pose = mp_pose.Pose(model_complexity=2, smooth_landmarks=True, enable_segmentation=True, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)    #better and more optimized for video analysis
#if Mediapipe holistic 
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(model_complexity=2, smooth_landmarks=True, enable_segmentation=True, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    ((24, 23, 11), None, ['body']),  # anca dx, anca sx, spalla sx


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
    

# Definizione dell'array angles_mirroring
angle_mirroring = [

    #<================================BRACCIA===============================>
    ((11, 13, 15), (12, 14, 16)),

    #<==================================GAMBA=================================>
    ((23, 25, 27), (24, 26, 28)),

    #<======================================CORPO===========================>
    ((12, 11, 23), (11, 12, 24)),
    ((24, 23, 11), (23, 24, 12)),

    #<==========================MANI==========================>
    ((15, 17, 19), (16, 18, 20)),
    ((15, 19, 21), (16, 20, 22)),
    #<==========================PIEDI==========================>
    ((27, 29, 31), (28, 30, 32)),

    #<==========================TESTA==========================>
    ((0, 1, 2), (0, 4, 5)),
    ((1, 2, 3), (4, 5, 6)),
    ((1, 0, 4), (1, 0, 4)),
    ((2, 3, 7), (5, 6, 8)),
    ((10, 9, 7), (9, 10, 8)),

    #<========================CORPO-BRACCIA======================================>
    ((23, 11, 13), (24, 12, 14)),
    ((12, 11, 13), (11, 12, 14)),

    #<========================CORPO-GAMBE======================================>
    ((11, 23, 25), (12, 24, 26)),
    ((24, 23, 25), (23, 24, 26)),
    
    #<========================MANI-BRACCIA======================================>
    ((13, 15, 17), (14, 16, 18)),
    ((13, 15, 21), (14, 16, 22)),
    #<========================PIEDI-GAMBE======================================>
    ((25, 27, 29), (26, 28, 30)),
    ((25, 27, 31), (26, 28, 32)),
]    

# Imposta la tolleranza per gli angoli e la soglia
ANGLE_TOLERANCE = 15  # in gradi
KEYPOINTS_CONFIDENCE_TOLERANCE= 0.5
ANGLES_CONFIDENCE_TOLERANCE= 0.5
ANGLE_CONFIDENCE_TOLERANCE= 0.1
SIMILARITY_THRESHOLD = 10

ANGLE_ERROR_TOLERANCE_THRESHOLD=20
ANGLE_WARNING_TOLERANCE_THRESHOLD=15

num_frames_to_check = 4
NUM_FRAMES_PASSED = 5
FIRST_FRAME_NUMBER = 25

DTW_REALTIME_FRAME_NUMBER = 5