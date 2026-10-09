from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AnomalyRequest(BaseModel):
    site_id: str
    as_of: Optional[str] = Field(None, description="ISO timestamp; defaults to latest available")


class ForecastRequest(BaseModel):
    site_id: str
    as_of: Optional[str] = None


class BatchRequest(BaseModel):
    site_ids: list[str]
    as_of: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 3


class AgentQueryRequest(BaseModel):
    question: str
