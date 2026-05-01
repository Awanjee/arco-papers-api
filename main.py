from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import chat
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Arco Papers AI Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (our HTML frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store — simple dict keyed by session_id
# In production this would be Redis or a database
sessions: dict = {}


class MessageRequest(BaseModel):
    message: str
    session_id: str = "default"


class MessageResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: MessageRequest):
    # Get or create session history
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    history = sessions[request.session_id]
    answer, updated_history = chat(request.message, history)
    sessions[request.session_id] = updated_history

    return MessageResponse(response=answer, session_id=request.session_id)


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}
