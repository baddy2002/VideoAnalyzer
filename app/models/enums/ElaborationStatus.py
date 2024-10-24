from enum import Enum


class ElaborationStatus(Enum):
    CREATED = 'CREATED'
    PROCESSING = 'PROCESSING'
    DELETED = 'DELETED'
    SAVED = 'SAVED'