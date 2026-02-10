from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from .models import *
from .storage.vector_storage import VectorStorage
from .storage.graph_storage import GraphStorage


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vector_storage = VectorStorage()
    app.state.graph_storage = GraphStorage()
    yield
    app.state.graph_storage.close()


app = FastAPI(lifespan=lifespan)


@app.post("/upload", response_model=List[DocumentUploadResponse])
async def upload_to_both_dbs(
    user_id: str = Form(...),
    document_name: str = Form(...),
    file: UploadFile = File(...),
    tags: str = Form(""),
    description: Optional[str] = Form(None)
):
    responses = []

    content = await file.read()
    text_content = content.decode('utf-8')
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    metadata = {
        "tags": tag_list,
        "description": description,
        "original_filename": file.filename,
        "content_type": file.content_type
    }

    try:
        vector_result = app.state.vector_storage.add_document(
            user_id=user_id,
            document_name=document_name,
            content=text_content,
            metadata=metadata
        )
        responses.append(DocumentUploadResponse(
            document_id=vector_result["document_id"],
            user_id=user_id,
            kb_type=KnowledgeBaseType.VECTOR,
            status="success",
            message="document added to vector database",
            timestamp=datetime.now(),
            chunks_processed=vector_result["chunks_processed"]
        ))
    except Exception as e:
        responses.append(DocumentUploadResponse(
            document_id="error",
            user_id=user_id,
            kb_type=KnowledgeBaseType.VECTOR,
            status="error",
            message=str(e),
            timestamp=datetime.now()
        ))

    try:
        graph_result = app.state.graph_storage.add_document(
            user_id=user_id,
            document_name=document_name,
            content=text_content,
            metadata=metadata
        )
        responses.append(DocumentUploadResponse(
            document_id=graph_result["document_id"],
            user_id=user_id,
            kb_type=KnowledgeBaseType.GRAPH,
            status="success",
            message="document added to graph database",
            timestamp=datetime.now(),
            entities_extracted=graph_result["entities_extracted"]
        ))
    except Exception as e:
        responses.append(DocumentUploadResponse(
            document_id="error",
            user_id=user_id,
            kb_type=KnowledgeBaseType.GRAPH,
            status="error",
            message=str(e),
            timestamp=datetime.now()
        ))

    return responses


@app.get("/query/vector")
async def query_vector_db(user_id: str, query: str, top_k: int = 5):
    results = app.state.vector_storage.query(user_id, query, top_k)
    return {
        "user_id": user_id,
        "query": query,
        "results_count": len(results),
        "results": results
    }


@app.get("/query/graph")
async def query_graph_db(user_id: str, query: str):
    results = app.state.graph_storage.query(user_id, query)
    return {
        "user_id": user_id,
        "query": query,
        "results_count": len(results),
        "results": results
    }


@app.get("/")
async def root():
    return {"message": "RAG API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_storage": hasattr(app.state, 'vector_storage'),
        "graph_storage": hasattr(app.state, 'graph_storage')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
