from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Demo API")

OPEN_ROUTER_KEY = os.environ.get("OPEN_ROUTER_KEY")
MODEL_ID = "arcee-ai/trinity-large-preview:free"


class QueryRequest(BaseModel):
    query: str
    max_tokens: int = 500
    temperature: float = 0.7


@app.post("/query")
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
