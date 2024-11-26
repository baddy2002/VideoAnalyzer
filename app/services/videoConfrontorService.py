import logging
import math
import os
from sqlite3 import DatabaseError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from sqlalchemy.exc import IntegrityError, OperationalError, DatabaseError, ProgrammingError
import shutil
from sqlalchemy.future import select

from app.CNN.frameConfrontator.posesConfrontation import frame_confrontation
from app.config.database import get_session
from app.config.settings import Settings
from app.managements import prefixs
from app.models.entities import ElaborationFrames, FrameAngle, Video
from uuid import UUID

from app.models.enums.ElaborationStatus import ElaborationStatus
from app.services.TransactionalDbService import save_elaboration
from app.services.movesDesignerService import create_elaboration_video, create_frame_image
settings = Settings()
logger = logging.getLogger(__name__)


def check_connection(   pose_connections, 
                        keypoints, 
                        total_min_x, 
                        min_x_key,
                        total_max_x,
                        max_x_key,
                        total_min_y, 
                        min_y_key,
                        total_max_y,
                        max_y_key,
                        delta_barycenter_x,
                        delta_barycenter_y,
                        eps=0.05):
    all_green = True
    value = abs(keypoints[min_x_key]['x'] - total_min_x)-delta_barycenter_x
    logger.info(f'value min x: {value}')
    if -eps < value and value > eps:
        return False, all_green
    value = abs(keypoints[min_y_key]['y'] - total_min_y)-delta_barycenter_y
    logger.info(f'value min y: {value}')
    if -eps < value and value > eps:
        return False, all_green
    value = abs(keypoints[max_x_key]['x'] - total_max_x)-delta_barycenter_x
    logger.info(f'confronting max_x: {value}')
    if -eps < value and value > eps:
        return False, all_green
    value= abs(keypoints[max_y_key]['y'] - total_max_y)-delta_barycenter_y
    logger.info(f'confronting max y : {value}')
    if -eps < value and value > eps:
        return False, all_green    

    for conn in pose_connections:
        if conn['color'] != '#00FF00' and conn['color'] != '#99FF00':
            all_green=False

    return True, all_green
    

# Funzione per normalizzare una connessione (ordinare gli indici)
def normalize_connection(connection):
    return tuple(sorted(connection))



def update_connection_color_and_difference(connection1, connection2):
    average_color = math.ceil((color_priority[connection1['color']] + color_priority[connection2['color']]) / 2)
    new_color = priority_color[average_color]        
    new_diff = connection1['diff']+ connection2['diff'] /2.0
    return new_color, new_diff



color_priority = {
    "#00FF00": 0,  # Green
    "#99FF00": 1,  # Green-Yellow (via di mezzo)
    "#FFFF00": 2,  # Yellow
    "#FFA500": 3,  # Orange
    "#FF0000": 4   # Red
}
priority_color = {v: k for k, v in color_priority.items()}
