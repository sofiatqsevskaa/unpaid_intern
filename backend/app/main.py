from logger import get_logger
import time
from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from storage_repository import StorageRepository
from models import *
from prompt_service import router as prompt_router

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    repo = StorageRepository()
    app.state.repo = repo
    yield
    repo.close()


app = FastAPI(lifespan=lifespan)
app.include_router(prompt_router)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} {response.status_code} {duration:.1f}ms")
    return response


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
    logger.info(f"Received upload: {file.filename} from user {user_id}")

    metadata = {
        "tags": tag_list,
        "description": description or "",
        "original_filename": file.filename,
        "content_type": file.content_type or "text/plain"
    }

    try:
        vector_result = app.state.repo.add_to_vector(
            user_id=user_id,
            document_name=document_name,
            content=text_content,
            metadata=metadata
        )
        if vector_result.get("skipped"):
            responses.append(DocumentUploadResponse(
                document_id="skipped",
                user_id=user_id,
                kb_type=KnowledgeBaseType.VECTOR,
                status="skipped",
                message=f"duplicate: {vector_result.get('reason')}",
                timestamp=datetime.now(),
                chunks_processed=0
            ))
        else:
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
        logger.error(
            f"vector upload failed for '{document_name}': {e}", exc_info=True)
        responses.append(DocumentUploadResponse(
            document_id="error",
            user_id=user_id,
            kb_type=KnowledgeBaseType.VECTOR,
            status="error",
            message=str(e),
            timestamp=datetime.now()
        ))

    try:
        graph_result = app.state.repo.add_to_graph(
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
        logger.error(
            f"graph upload failed for '{document_name}': {e}", exc_info=True)
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
    results = app.state.repo.query_vector(user_id, query, top_k)
    return {"user_id": user_id, "query": query, "results_count": len(results), "results": results}


@app.get("/query/graph")
async def query_graph_db(user_id: str, query: str):
    results = app.state.repo.query_graph(user_id, query)
    return {"user_id": user_id, "query": query, "results_count": len(results), "results": results}


@app.get("/")
async def root():
    return {"message": "Multi-KB RAG API is running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_storage": hasattr(app.state, 'repo') and app.state.repo.vector is not None,
        "graph_storage": hasattr(app.state, 'repo') and app.state.repo.graph is not None
    }


@app.get("/list_documents")
async def list_documents(user_id: str):
    vector_docs = app.state.repo.vector.list_documents(user_id)
    graph_docs = app.state.repo.graph.list_documents(user_id)
    return {
        "user_id": user_id,
        "vector_documents": vector_docs,
        "graph_documents": graph_docs
    }
