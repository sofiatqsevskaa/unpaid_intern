from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class KnowledgeBaseType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class DocumentUploadRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    document_name: str = Field(..., description="Name of the document")
    content: Optional[str] = Field(None, description="Direct text content")
    file_path: Optional[str] = Field(None, description="Path to uploaded file")
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: str
    user_id: str
    kb_type: KnowledgeBaseType
    status: str
    message: str
    timestamp: datetime
    chunks_processed: Optional[int] = None
    entities_extracted: Optional[int] = None
