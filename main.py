# HOW TO RUN
# ----------
# 1. Add your API keys to the .env file:
#       GROQ_API_KEY=gsk_...
#       LANGSMITH_API_KEY=lsv2_...   (optional)
#
# 2. Install dependencies:
#       pip install -r requirements.txt
#
# 3. Start the server:
#       python main.py
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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")

if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "agricultural-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY not set in .env")
    else:
        print("Agricultural AI Agent started. LangSmith:", "enabled" if LANGSMITH_API_KEY else "disabled")
    yield


app = FastAPI(
    title="Agricultural AI Agent",
    description="AI agent with Pest Agent and Scheme Agent tools, powered by Groq and traced via LangSmith.",
    version="1.0.0",
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
    message: str


class ChatResponse(BaseModel):
    response: str
    tools_used: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    return HTMLResponse(content=UI_HTML)


@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "healthy",
        "service": "Agricultural AI Agent",
        "groq_configured": bool(GROQ_API_KEY),
        "langsmith_enabled": bool(LANGSMITH_API_KEY),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a message to the Agricultural AI Agent.

    Routes automatically to:
    - **pest_agent** for pest-related queries
    - **scheme_agent** for government scheme / subsidy queries
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in .env file.")
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty.")

    try:
        result = run_agent(
            message=request.message,
            groq_api_key=GROQ_API_KEY,
            langsmith_api_key=LANGSMITH_API_KEY or None,
        )
        return ChatResponse(response=result["response"], tools_used=result["tools_used"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Embedded HTML UI ───────────────────────────────────────────────────────────

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Agricultural AI Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f4f0; color: #222; }
    header { background: #2d6a4f; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 1.3rem; }
    .container { max-width: 860px; margin: 28px auto; padding: 0 16px; }

    .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
    .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 14px; color: #2d6a4f; }

    textarea {
      width: 100%; padding: 10px 13px; border: 1px solid #ccc; border-radius: 6px;
      font-size: .92rem; resize: vertical; min-height: 80px; font-family: inherit;
    }
    textarea:focus { outline: none; border-color: #2d6a4f; }

    button {
      margin-top: 12px; padding: 10px 26px; background: #2d6a4f; color: white;
      border: none; border-radius: 6px; font-size: .92rem; cursor: pointer; font-weight: 600;
    }
    button:hover { background: #1b4332; }
    button:disabled { background: #aaa; cursor: not-allowed; }

    #chat-history { max-height: 500px; overflow-y: auto; margin-bottom: 14px; }
    .bubble { padding: 12px 16px; border-radius: 10px; margin-bottom: 10px; line-height: 1.6; font-size: .91rem; }
    .user-bubble  { background: #d8f3dc; margin-left: 60px; text-align: right; }
    .agent-bubble { background: #f8f9fa; border: 1px solid #e2e8e2; margin-right: 60px; white-space: pre-wrap; }
    .tools-tag { margin-top: 8px; font-size: .75rem; color: #888; }
    .tools-tag span { background: #eef2ee; padding: 2px 9px; border-radius: 20px; margin-right: 4px; font-weight: 500; }
    .loading { color: #999; font-style: italic; }
    .err-bubble { color: #c0392b; }

    .hint { font-size: .8rem; color: #777; margin-top: 8px; }
    .hint b { color: #2d6a4f; }
    .status-bar { font-size: .8rem; padding: 7px 12px; border-radius: 6px; margin-bottom: 16px; }
    .status-ok  { background: #d8f3dc; color: #1b4332; }
    .status-err { background: #fdecea; color: #c0392b; }
  </style>
</head>
<body>
<header>
  <svg width="26" height="26" fill="white" viewBox="0 0 24 24">
    <path d="M17 8C8 10 5.9 16.17 3.82 20H5.71C6.77 18 8 16.44 9.67 15.17C12 17 15 17 18 16C20 15.38 21.5 13.5 22 12C19.33 13.33 17.67 13 17 13C19 12 20.5 9 20.5 7C18.67 8 17.67 7.67 17 8Z"/>
  </svg>
  <h1>Agricultural AI Agent</h1>
</header>

<div class="container">
  <div id="statusBar"></div>

  <div class="card">
    <h2>Chat with the Agent</h2>
    <div id="chat-history"></div>
    <textarea id="userMsg"
      placeholder="Ask about pests or government schemes...&#10;e.g. How do I control aphids on tomato?&#10;e.g. What is PM-KISAN and how do I apply?"></textarea>
    <div class="hint">
      Automatically routes to <b>Pest Agent</b> or <b>Scheme Agent</b> based on your question.
    </div>
    <button id="sendBtn" onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
  async function checkHealth() {
    try {
      const r = await fetch('/health');
      const d = await r.json();
      const bar = document.getElementById('statusBar');
      if (!d.groq_configured) {
        bar.className = 'status-bar status-err';
        bar.textContent = '⚠ GROQ_API_KEY not set in .env — chats will fail.';
        document.getElementById('sendBtn').disabled = true;
      } else {
        bar.className = 'status-bar status-ok';
        bar.textContent = '✓ Connected — Groq Qwen3-32b' + (d.langsmith_enabled ? ' · LangSmith tracing active' : '');
      }
    } catch {}
  }
  checkHealth();

  async function sendMessage() {
    const box = document.getElementById('userMsg');
    const msg = box.value.trim();
    if (!msg) return;
    const hist = document.getElementById('chat-history');

    hist.innerHTML += `<div class="bubble user-bubble">${escHtml(msg)}</div>`;
    box.value = '';

    const loading = document.createElement('div');
    loading.className = 'bubble agent-bubble loading';
    loading.textContent = 'Thinking…';
    hist.appendChild(loading);
    hist.scrollTop = hist.scrollHeight;

    document.getElementById('sendBtn').disabled = true;

    try {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: msg })
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail);

      loading.className = 'bubble agent-bubble';
      loading.innerHTML = escHtml(d.response) +
        (d.tools_used.length
          ? `<div class="tools-tag">via: ${d.tools_used.map(t => `<span>${t}</span>`).join('')}</div>`
          : '');
    } catch (e) {
      loading.className = 'bubble agent-bubble err-bubble';
      loading.textContent = 'Error: ' + e.message;
    }

    document.getElementById('sendBtn').disabled = false;
    hist.scrollTop = hist.scrollHeight;
  }

  document.getElementById('userMsg').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
  }
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
