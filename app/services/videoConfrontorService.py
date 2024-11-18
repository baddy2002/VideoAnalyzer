import logging
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
from app.managements.mediapipe import FIRST_FRAME_NUMBER, num_frames_to_check
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
                        eps=0.1):
    all_green = True

    logger.info(f'confronting min x: {keypoints[min_x_key]['x']} con: {total_min_x-delta_barycenter_x}')
    if abs(keypoints[min_x_key]['x'] - total_min_x-delta_barycenter_x) > eps:
        return False, all_green
    logger.info(f'confronting min y: {keypoints[min_y_key]['y']} con: { total_min_y-delta_barycenter_y}')
    if abs(keypoints[min_y_key]['y'] - total_min_y-delta_barycenter_y) > eps:
        return False, all_green
    
    logger.info(f'confronting max_x: {keypoints[max_x_key]['x']} con: {total_max_x+delta_barycenter_x}')
    if abs(keypoints[max_x_key]['x'] - total_max_x+delta_barycenter_x) > eps:
        return False, all_green
    logger.info(f'confronting max y : {keypoints[max_y_key]['y']} con: {total_max_y+delta_barycenter_y}')
    if abs(keypoints[max_y_key]['y'] - total_max_y+delta_barycenter_y) > eps:
        return False, all_green    

    for conn in pose_connections:
        if conn['color'] != '#00FF00':
            all_green=False

    return True, all_green
    
