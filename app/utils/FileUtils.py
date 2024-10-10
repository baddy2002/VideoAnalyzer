import mimetypes
import magic

class FileUtils:
    @staticmethod
    async def get_mime_type_with_name(file):
        mime_type, _ = mimetypes.guess_type(file.filename)
        return mime_type

    @staticmethod
    async def get_mime_type(file):
        # leggi contenuto
        file_content = await file.read()
        # Crea un oggetto magic
        mime = magic.Magic()
        # Determina il tipo MIME dal contenuto
        mime_type = mime.from_buffer(file_content)
        await file.seek(0)  # Riporta il puntatore del file all'inizio
        return mime_type
        
    @staticmethod
    async def get_extension_from_mime(mime_type):
        if "jpeg" in mime_type or "jpg" in mime_type:
            return ".jpg"
        elif "png" in mime_type:
            return ".png"
        elif "webp" in mime_type or "Web/P" in mime_type:
            return ".webp"
        # Aggiungi il supporto per i video
        elif "mp4" in mime_type:
            return ".mp4"
        elif "avi" in mime_type:
            return ".avi"
        elif "mov" in mime_type:
            return ".mov"
        elif "mkv" in mime_type:
            return ".mkv"
        return ""
        
    @staticmethod
    def get_fourcc_from_extension(extension):
        """
        Funzione di utilità per restituire il fourcc corretto in base all'estensione del file video.
        """
        if extension in ['.mp4', '.m4v']:
            return 'mp4v'  # Fourcc per MP4
        elif extension in ['.avi']:
            return 'XVID'  # Fourcc per AVI
        elif extension in ['.mov']:
            return 'avc1'  # Fourcc per MOV
        else:
            raise ValueError(f"Estensione video non supportata: {extension}")