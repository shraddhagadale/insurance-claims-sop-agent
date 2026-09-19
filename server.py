from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .sop_agent.config import PACKAGE_DIR, settings
from .sop_agent.engine import SOPAgent
from .sop_agent.state import Session


class SessionResponse(BaseModel):
    session_id: str
    phase: str
    reply: str
    verified: bool
    active_case_id: str | None
    resolved_intent: str | None
    email_status: str
    trace: dict[str, Any]


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)


class MessageResponse(SessionResponse):
    handoff: dict[str, Any] | None = None


class AppState:
    agent: SOPAgent | None = None
    sessions: dict[str, Session] = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.require_credentials()
    state.agent = SOPAgent(settings=settings, require_api_key=True)
    yield


app = FastAPI(title="Insurance Claims SOP Agent", lifespan=lifespan)

static_dir = PACKAGE_DIR / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    session = state.agent.new_session()
    state.sessions[session.id] = session
    reply = session.transcript[-1]["text"]
    return SessionResponse(
        session_id=session.id,
        phase=session.phase.value,
        reply=reply,
        verified=session.verification.verified,
        active_case_id=session.active_case_id,
        resolved_intent=session.resolved_intent,
        email_status=session.email_status,
        trace={"phase": session.phase.value, "awaiting": session.awaiting, "memory": session.memory.snapshot()},
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = _session(session_id)
    reply = session.transcript[-1]["text"] if session.transcript else ""
    return SessionResponse(
        session_id=session.id,
        phase=session.phase.value,
        reply=reply,
        verified=session.verification.verified,
        active_case_id=session.active_case_id,
        resolved_intent=session.resolved_intent,
        email_status=session.email_status,
        trace={"transcript": session.transcript, "audit": session.audit, "memory": session.memory.snapshot()},
    )


@app.post("/sessions/{session_id}/messages", response_model=MessageResponse)
def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent is not ready")
    session = _session(session_id)
    result = state.agent.handle_turn(session, request.message)
    return MessageResponse(**result.__dict__)


def _session(session_id: str) -> Session:
    session = state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")
    return session
