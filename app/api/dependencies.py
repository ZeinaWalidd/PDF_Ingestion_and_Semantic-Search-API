from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services.ingestion_service import IngestionService
from app.services.ports import ChunkRepository
from app.services.search_service import SearchService


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_repository(request: Request) -> ChunkRepository:
    return request.app.state.repository


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
IngestionServiceDep = Annotated[IngestionService, Depends(get_ingestion_service)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
RepositoryDep = Annotated[ChunkRepository, Depends(get_repository)]
