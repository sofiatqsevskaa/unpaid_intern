from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

OPEN_ROUTER_KEY = os.environ.get("OPEN_ROUTER_KEY")
MODEL_ID = "arcee-ai/trinity-large-preview:free"


class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 500
    temperature: float = 0.7


class RAGQueryRequest(BaseModel):
    query: str
    vector_results: Optional[List[Dict[str, Any]]] = []
    graph_results: Optional[List[Dict[str, Any]]] = []
    max_tokens: int = 500
    temperature: float = 0.3


@router.post("/query")
def query_model(req: QueryRequest):
    headers = {
        "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": req.query}
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
        return {"response": answer}

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling OpenRouter API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@router.post("/rag/query")
def rag_query(req: RAGQueryRequest):
    context_parts = []

    if req.vector_results:
        context_parts.append("DOCUMENT CHUNKS (VECTOR SEARCH)")
        for i, res in enumerate(req.vector_results, 1):
            doc_name = res.get('metadata', {}).get('document_name', 'unknown')
            content = res.get('content', "")
            context_parts.append(f"[Chunk {i} from {doc_name}]:\n{content}\n")
    if req.graph_results:
        context_parts.append(
            "ENTITIES AND RELATIONSHIPS (GRAPH SEARCH)")
        for i, res in enumerate(req.graph_results, 1):
            doc = res.get("document", {})
            doc_name = doc.get('name', 'unknown')
            context_parts.append(f"[Document {i}: {doc_name}]")

            entities = res.get("entities", [])
            if entities:
                entity_text = ", ".join(
                    [f"{e.get('name')} ({e.get('type')})" for e in entities])
                context_parts.append(f"Entities mentioned: {entity_text}")

            preview = doc.get('content_preview', '')
            if preview:
                context_parts.append(f"Content preview: {preview}")

            context_parts.append("")

    context = "\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Be concise and accurate. Always cite which document your information comes from."""

    user_prompt = f"""CONTEXT:
{context}

QUESTION: {req.query}

Answer based only on the context above:"""

    headers = {
        "Authorization": f"Bearer {OPEN_ROUTER_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        return {
            "response": answer,
            "context_used": {
                "vector_results_count": len(req.vector_results),
                "graph_results_count": len(req.graph_results)
            }
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500, detail=f"Error calling OpenRouter API: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
