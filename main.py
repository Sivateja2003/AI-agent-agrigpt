# HOW TO RUN
# ----------
# 1. Add your Groq API key to the .env file (used by /schemes and /pests endpoints):
#       GROQ_API_KEY=gsk_...
#
# 2. Install dependencies:
#       pip install -r requirements.txt
#
# 3. Start the server (local):
#       uvicorn main:app --reload
#
#    Deploy on Render:
#       Build Command  : pip install -r requirements.txt
#       Start Command  : gunicorn -k uvicorn.workers.UvicornWorker main:app
#
# 4. Open in browser:
#       http://localhost:8000
#
# API docs: http://localhost:8000/docs
# ----------

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import run_agent
from tools import query_pest_structured, query_scheme_structured

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")

_langsmith_active = bool(LANGCHAIN_API_KEY and not LANGCHAIN_API_KEY.startswith("lsv2_your"))
if _langsmith_active:
    os.environ["LANGCHAIN_API_KEY"]    = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "agricultural-agent")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set in .env (required for /schemes and /pests endpoints)")
    else:
        tracing = "LangSmith tracing enabled" if _langsmith_active else "LangSmith tracing disabled"
        print(f"AgriGPT AI Agent started with Groq. {tracing}.")
    yield


app = FastAPI(
    title="AgriGPT — Agricultural AI Agent",
    description="AI agent for farmers powered by Groq. Routes pest and government scheme queries to specialized tools.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    chatId: str
    phone_number: str
    message: str
    api_key: str


class ChatResponse(BaseModel):
    response: str
    sources: list[str]


class PestRequest(BaseModel):
    query: str


class PestResponse(BaseModel):
    pest: str
    details: str


class SchemeRequest(BaseModel):
    query: str


class SchemeResponse(BaseModel):
    scheme: str
    details: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "service": "AgriGPT Agricultural AI Agent",
        "gemini_configured": bool(GROQ_API_KEY),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the Agricultural AI Agent.

    Routes automatically to:
    - **simulate_pests** for pest/disease-related queries
    - **government_schemes** for government scheme / subsidy queries
    - No tool called for unrelated questions
    """
    if not request.api_key.strip():
        raise HTTPException(status_code=400, detail="api_key cannot be empty.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    try:
        result = run_agent(message=request.message, api_key=request.api_key)
        return ChatResponse(response=result["response"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pests", response_model=PestResponse, tags=["Pests"])
async def pests(request: PestRequest):
    """
    Query the pest and disease knowledge base directly.

    Returns the matched pest/disease name and detailed information
    including symptoms and control measures.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")

    try:
        result = query_pest_structured(api_key=GROQ_API_KEY, query=request.query)
        return PestResponse(pest=result.get("pest", "Unknown"), details=result.get("details", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/schemes", response_model=SchemeResponse, tags=["Schemes"])
async def schemes(request: SchemeRequest):
    """
    Query the government schemes knowledge base directly.

    Returns the matched scheme name and detailed information
    about eligibility, benefits, and how to apply.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty.")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")

    try:
        result = query_scheme_structured(api_key=GROQ_API_KEY, query=request.query)
        return SchemeResponse(scheme=result.get("scheme", "Unknown"), details=result.get("details", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
