from app.managements import prefixs
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from app.models.requests import analyzeRequests
from app.CNN.frameAnalyzer import analyze
from fastapi.responses import StreamingResponse
import logging
from utils.FileUtils import FileUtils

logger = logging.getLogger(__name__)

image_analysis_router = APIRouter(
    prefix=prefixs.analyze_prefix,
    tags=["Image confrontation"],
    responses={404: {"description": "Not found"}},
)

@image_analysis_router.post("/")
async def analyze_images(
        area: str = Form(...),     # Accetta come stringa JSON
    portions: str = Form(...),  # Accetta come stringa JSON
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    # Carica i dati JSON nei modelli
    try:
        area_data = analyzeRequests.Area.parse_raw(area)
        portions_data = analyzeRequests.Portions.parse_raw(portions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


    if round(area_data.dx + area_data.sx, 1) != 1.0:
        raise HTTPException(status_code=400, detail="La somma dei valori di area deve essere 100%.")

    # Controlla che la somma delle porzioni sia 100%
    total_portions = sum(portions_data.dict().values())
    if round(total_portions, 1) != 1.0:
        raise HTTPException(status_code=400, detail="La somma dei valori di portions deve essere 100%.")


    # Ottieni le estensioni dei file
    extension1 = await FileUtils.get_mime_type_with_name(file1)
    extension2 = await FileUtils.get_mime_type_with_name(file2)
    if(extension1 is None):                                     #cerca di evitare di dover leggere il contenuto per risalire al tipo
        extension1 = await FileUtils.get_mime_type(file1)
    if(extension2 is None):
        extension2 = await FileUtils.get_mime_type(file2)
         
    extension1 = await FileUtils.get_extension_from_mime(extension1)
    extension2 = await FileUtils.get_extension_from_mime(extension2)
    
    similar, stream1, stream2 = await analyze.single_frame_confrontation(file1, file2, extension1, extension2, area_data.dict(), portions_data.dict())
    response = StreamingResponse(stream2, media_type=file2.content_type)
    response.headers['Content-Disposition'] = f'attachment; filename="{file2.filename}"'
    response.headers['X-similar'] = str(similar).lower() 
    return response

