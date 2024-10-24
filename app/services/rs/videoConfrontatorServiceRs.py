from sqlalchemy import func
from app.config.settings import Settings
from app.managements import prefixs
from fastapi import APIRouter, File, Request, Response, UploadFile, HTTPException, Form, BackgroundTasks, Query, Path, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import logging
import shutil
from tempfile import NamedTemporaryFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Optional, List
from app.models.requests import analyzeRequests
from app.models.entities import Elaboration, Video, FrameAngle
from app.models.responses import VideoResponse
from app.CNN.frameAnalyzer import analyze
from utils.FileUtils import FileUtils
import os
from uuid import UUID, uuid4
from app.config.database import get_session

settings = Settings()
logger = logging.getLogger(__name__)

video_confrontation_router = APIRouter(
    prefix=prefixs.confrontation_prefix,
    tags=["Video confrontations "],
    responses={404: {"description": "URL Not found"}},
)

@video_confrontation_router.get("/{video_uuid}", response_model=List[VideoResponse.Elaboration])
async def get_analysis(
    video_uuid: UUID = Path(..., description="UUID del video da recuperare"),  # Path parameter
    startRow: int = Query(0, ge=0),  
    pageSize: int = Query(10, gt=0),  
    order_by: str = Query("created_at DESC"),  # Ordinamento predefinito
    obj_status: str =  Query('SAVED')
):
    
    
    async with get_session() as session:
        # Inizia la query di base
        query = select(Elaboration.Elaboration).where(Elaboration.Elaboration.video_uuid == str(video_uuid))
        count_query = select(func.count()).select_from(Elaboration.Elaboration).where(Elaboration.Elaboration.video_uuid == str(video_uuid))


        # Applica l'ordinamento
        if order_by:
            order_by_clause = order_by.split()
            column_name = order_by_clause[0]
            direction = order_by_clause[1] if len(order_by_clause) > 1 else "ASC"
            if direction.upper() == "DESC":
                query = query.order_by(getattr(Elaboration.Elaboration, column_name).desc())
            else:
                query = query.order_by(getattr(Elaboration.Elaboration, column_name))

        if obj_status:
            count_query = count_query.where(Elaboration.Elaboration.status == obj_status.upper())
            query = query.where(Elaboration.Elaboration.status == obj_status.upper())

        # Esegui la query con paginazione
        query = query.offset(startRow).limit(pageSize)
        total_result = await session.execute(count_query)
        total_count = total_result.scalar()  # Assicurati di ottenere un numero intero
        # Esegui la query finale
        result = await session.execute(query)
        elaborations = result.scalars().all()
        elaboration_list = [{
            "uuid": str(elaboration.uuid),
            "name": elaboration.name,
            "format": elaboration.format,
            "size": elaboration.size,
            "thumbnail": elaboration.thumbnail,
            "video_uuid": elaboration.video_uuid,
        } for elaboration in elaborations]

        response = JSONResponse(content=elaboration_list)
        response.headers["listsize"] = str(total_count)
        response.headers["Access-Control-Expose-Headers"] = "listsize"
        return response


@video_confrontation_router.get('/{video_uuid}/{uuid}', response_model=VideoResponse.Elaboration)
async def get_single_video(
    video_uuid: UUID = Path(..., description="UUID del video da recuperare"),
    uuid: UUID = Path(..., description="UUID dell'elaborazione specifica")  # Path parameter
):
    try:
        async with get_session() as session:
            # Costruisci la query per cercare un singolo video
            query = select(Elaboration.Elaboration).where(Elaboration.Elaboration.uuid == uuid)
            
            # Esegui la query
            result = await session.execute(query)
            elaboration = result.scalars().first()
            
            # Se non viene trovato il video, ritorna un errore 404
            if elaboration is None:
                return VideoResponse.JsonObjectNotFoundResponse(entity="Elaboration", uuid=uuid)
            
            return elaboration
    except Exception as e:
        return VideoResponse.JsonServerErrorResponse('errore durante la creazione della risposta  '+ str(e))


@video_confrontation_router.get('/a/{video_filename}/streaming')
async def get_video_stream(request: Request, video_filename: str):
    video_path = os.path.join(settings.VIDEO_FOLDER, video_filename)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    file_size = os.path.getsize(video_path)
    headers = {}

    range_header = request.headers.get('range')
    if range_header:
        range_value = range_header.strip().replace('bytes=', '')
        if not range_value.__contains__('-'):
            return VideoResponse.JsonBadRequestResponse("header range malformed")
        range_start, range_end = range_value.split('-')

        range_start = int(range_start)
        range_end = range_end or file_size - 1
        range_end = int(range_end)

        if range_start >= file_size or range_end >= file_size:
            raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")

        content_length = (range_end - range_start) + 1
        headers['Content-Range'] = f'bytes {range_start}-{range_end}/{file_size}'
        headers['Accept-Ranges'] = 'bytes'
        headers['Content-Length'] = str(content_length)
        headers['Access-Control-Allow-Origin'] = '*'
        headers['Cache-Control'] = 'no-cache'
        headers['Content-Encoding'] = 'identity'
        headers['Access-Control-Expose-Headers'] = 'Content-Range, Accept-Ranges, Content-Length, Access-Control-Allow-Origin, Content-Encoding'
        def iterfile():
            with open(video_path, 'rb') as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data
                    remaining -= len(data)

        return StreamingResponse(iterfile(), headers=headers, status_code=206, media_type="video/mp4")
    headers={
        'Content-Disposition': 'attachment; filename=video.mp4'
    }
    return FileResponse(video_path, headers=headers, media_type='video/mp4')
