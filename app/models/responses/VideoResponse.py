from datetime import datetime
from typing import List
from pydantic import BaseModel
from uuid import UUID
from fastapi.responses import JSONResponse

class JsonResponseModel(JSONResponse):
    def __init__(self, message: str, status_code=200, additional_fields=None):
        content = {"message": message}
        if additional_fields:
            content.update(additional_fields)
        super().__init__(status_code=status_code, content=content)


class JsonBadRequestResponse(JsonResponseModel):

    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)

class JsonServerErrorResponse(JsonResponseModel):

    def __init__(self, message: str):
        super().__init__(message=message, status_code=500)

class JsonObjectNotFoundResponse(JsonResponseModel):

    def __init__(self, entity: str, uuid: str):
        super().__init__(message=f'Object: {entity} with uuid: {uuid} not found', status_code=404)

class Video(BaseModel, JSONResponse):
    uuid: UUID
    name: str
    format: str
    size: int
    description: str
    thumbnail: str
    area: dict
    portions: dict
    fps: float
    width: float
    height: float

    class Config:
        orm_mode = True  # Permette a Pydantic di lavorare con SQLAlchemy
        from_attributes=True



class Elaboration(BaseModel, JSONResponse):
    uuid: UUID
    name: str
    format: str
    size: int
    thumbnail: str
    video_uuid: str

    class Config:
        orm_mode = True  # Permette a Pydantic di lavorare con SQLAlchemy
        from_attributes=True


class ElaborationFrame(BaseModel, JSONResponse):
    uuid: UUID
    frame_number: int
    keypoints: list
    connections: list
    elaboration_uuid: UUID
    correct_keypoints: list

    class Config:
        orm_mode = True  # Permette a Pydantic di lavorare con SQLAlchemy
        from_attributes=True