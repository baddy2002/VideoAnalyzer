from pydantic import BaseModel, Field
from fastapi import File, UploadFile
from typing import Annotated

class Area(BaseModel):
    dx: float = Field(..., ge=0, le=1)
    sx: float = Field(..., ge=0, le=1)

class Portions(BaseModel):
    head: float = Field(..., ge=0, le=1)
    body: float = Field(..., ge=0, le=1)
    feets: float = Field(..., ge=0, le=1)
    hands: float = Field(..., ge=0, le=1)
    arms: float = Field(..., ge=0, le=1)
    legs: float = Field(..., ge=0, le=1)

class ImagesAnalysisRequest(BaseModel):
    area: Area
    portions: Portions
