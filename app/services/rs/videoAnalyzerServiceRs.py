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
from app.models.entities import Video, FrameAngle
from app.models.responses import VideoResponse
from app.CNN.frameAnalyzer import analyze
from utils.FileUtils import FileUtils
import os
from uuid import UUID
from app.config.database import get_session

settings = Settings()
logger = logging.getLogger(__name__)

video_analysis_router = APIRouter(
    prefix=prefixs.analyze_prefix,
    tags=["Video analysis"],
    responses={404: {"description": "URL Not found"}},
)

@video_analysis_router.get("/", response_model=List[VideoResponse.Video])
async def get_analysis(
    startRow: int = Query(0, ge=0),  # Inizio della paginazione
    pageSize: int = Query(10, gt=0),  # Numero di elementi per pagina
    order_by: str = Query("created_at DESC"),  # Ordinamento predefinito
    like_name: Optional[str] = Query(None)  # Filtro facoltativo per LIKE
):
    async with get_session() as session:
        # Inizia la query di base
        query = select(Video.Video)
        count_query = select(func.count()).select_from(Video.Video)

        # Applica il filtro LIKE se fornito
        if like_name:
            count_query = count_query.where(Video.Video.name.ilike(f"%{like_name}%"))
            query = query.where(Video.Video.name.ilike(f"%{like_name}%"))

        # Applica l'ordinamento
        if order_by:
            order_by_clause = order_by.split()
            column_name = order_by_clause[0]
            direction = order_by_clause[1] if len(order_by_clause) > 1 else "ASC"
            if direction.upper() == "DESC":
                query = query.order_by(getattr(Video.Video, column_name).desc())
            else:
                query = query.order_by(getattr(Video.Video, column_name))

        # Esegui la query con paginazione
        query = query.offset(startRow).limit(pageSize)
        total_result = await session.execute(count_query)
        total_count = total_result.scalar()  # Assicurati di ottenere un numero intero
        # Esegui la query finale
        result = await session.execute(query)
        videos = result.scalars().all()
        video_list = [{
            "uuid": str(video.uuid),
            "name": video.name,
            "format": video.format,
            "size": video.size,
            "description": video.description,
            "thumbnail": video.thumbnail,
        } for video in videos]

        response = JSONResponse(content=video_list)
        response.headers["listsize"] = str(total_count)
        response.headers["Access-Control-Expose-Headers"] = "listsize"
        return response


@video_analysis_router.get('/{uuid}', response_model=VideoResponse.Video)
async def get_single_video(
    uuid: UUID = Path(..., description="UUID del video da recuperare")  # Path parameter
):
    try:
        async with get_session() as session:
            # Costruisci la query per cercare un singolo video
            query = select(Video.Video).where(Video.Video.uuid == uuid)
            
            # Esegui la query
            result = await session.execute(query)
            video = result.scalars().first()
            
            # Se non viene trovato il video, ritorna un errore 404
            if video is None:
                return VideoResponse.JsonObjectNotFoundResponse(entity="Video", uuid=uuid)
            
            return video
    except Exception as e:
        return VideoResponse.JsonServerErrorResponse('errore durante la creazione della risposta  '+ str(e))


@video_analysis_router.get('/{video_filename}/streaming')
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
        logger.info("here we are")
        def iterfile():
            with open(video_path, 'rb') as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    logger.info("video data: " + str(data))
                    yield data
                    remaining -= len(data)

        return StreamingResponse(iterfile(), headers=headers, status_code=206, media_type="video/mp4")
    headers={
        'Content-Disposition': 'attachment; filename=video.mp4'
    }
    return FileResponse(video_path, headers=headers, media_type='video/mp4')

@video_analysis_router.post("/")
async def analyze_videos(
    background_tasks: BackgroundTasks,
    area: str = Form(...),     # Accetta come stringa 
    portions: str = Form(...),  
    description: str = Form(...),
    file1: UploadFile = File(...),      #File "corretto" che verrà usato per il confronto
):
    if not isinstance(description, str):
        VideoResponse.JsonBadRequestResponse("La descrizione deve essere una stringa valida.")
    elif not description or len(description) < 5 or description.isspace():
        VideoResponse.JsonBadRequestResponse("La descrizione non può essere vuota o minore di 5 caratteri.")
    # Carica i dati JSON nei modelli
    try:
        area_data = analyzeRequests.Area.parse_raw(area)
        portions_data = analyzeRequests.Portions.parse_raw(portions)
    except ValueError as e:
        logger.error(f"Errore durante il parsing dei JSON: {e}")
        return VideoResponse.JsonBadRequestResponse(f"Errore durante il parsing: {e}")

    total_data = sum(area_data.dict().values())
    if round(total_data, 1) > 1.0 or round(total_data, 1) < 1.0:
        return VideoResponse.JsonBadRequestResponse("La somma dei valori di area deve essere 100%. have: " + str(total_data*100))

    # Controlla che la somma delle porzioni sia 100%
    total_portions = sum(portions_data.dict().values())
    if round(total_portions, 1) > 1.0 or round(total_portions, 1) < 1.0:
        return VideoResponse.JsonBadRequestResponse("La somma dei valori di portions deve essere 100%. have: " + str(total_portions*100))


    # Ottieni le estensioni dei file
    extension1 = await FileUtils.get_mime_type_with_name(file1)
    if(extension1 is None):                                     #cerca di evitare di dover leggere il contenuto per risalire al tipo
        extension1 = await FileUtils.get_mime_type(file1)

    extension1 = await FileUtils.get_extension_from_mime(extension1)

    # Crea un file temporaneo per salvare il video
    temp_file = NamedTemporaryFile(delete=False, suffix=extension1, dir="/usr/srv")
    with temp_file as tmp:
        shutil.copyfileobj(file1.file, tmp)
    video_name=file1.filename
    logger.info("File caricato correttamente." + str(temp_file.name))

    logger.info("...begin analysis...")
    background_tasks.add_task(analyze.analyze_video_frames, temp_file, extension1, area_data.dict(), portions_data.dict(), video_name, description)
    logger.info("...end analysis...")

    
    return VideoResponse.JsonResponseModel(
        "request processed, the video can require some minutes to be reachable, "+
        "if after some minutes you cannot see the video you can retry to send it")


@video_analysis_router.get('/{uuid}/keypoints')
async def get_video_keypoints(
    uuid: UUID = Path(..., description="UUID del video da recuperare")  # Path parameter
):
    try:
        async with get_session() as session:
            # Costruisci la query per cercare un singolo video
            query = select(FrameAngle.FrameAngle).where(FrameAngle.FrameAngle.video_uuid == str(uuid))
            
            # Esegui la query
            result = await session.execute(query)
            video_analysis = result.scalars().all()
            
            # Se non viene trovato il video, ritorna un errore 404
            if video_analysis is None:
                return VideoResponse.JsonObjectNotFoundResponse(entity="Video", uuid=uuid)
            
            response = {frame_analysis.frame_number: frame_analysis.keypoints for frame_analysis in video_analysis}
            return JSONResponse(status_code=200, content=response)
    except Exception as e:
        return VideoResponse.JsonServerErrorResponse('errore durante la creazione della risposta  '+ str(e))


#<-------------------------------------------FUTURE STREAM-------------------------------------------->


@video_analysis_router.websocket("/ws/video_stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        # Riceve frame dal frontend
        frame_bytes = await websocket.receive_bytes()

        # TODO: Prendi i dati e usali per la nuova funzione del confronto
        # TODO: nuova funzione di confronto

        # Invia frame elaborato di nuovo al frontend
        await websocket.send_bytes(frame_bytes)