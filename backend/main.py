import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

load_dotenv()

from agents import run_chat_agent, run_extractor_agent, run_summarizer_agent
from database import Base, engine, get_db
from models import ChatHistory, Report, User

UPLOAD_DIR = Path("uploaded_reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Rectal MRI V2 Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    report_id: int
    user_id: int
    extracted_json: dict
    summary: str


class ReportItem(BaseModel):
    id: int
    source_filename: str | None
    report_summary: str | None
    t_stage: str | None
    n_stage: str | None
    crm_status: str | None
    emvi_status: str | None
    mrtrg_score: str | None
    tumor_deposits: str | None
    created_at: datetime | None


class ChatHistoryItem(BaseModel):
    id: int
    role: str
    message: str
    created_at: datetime | None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    model_choice: Literal["Gemini", "ZhipuAI", "OpenAI"] = "Gemini"


class ChatResponse(BaseModel):
    report_id: int
    answer: str
    metadata: dict


def _get_or_create_user(db: Session, requested_user_id: int | None = None) -> User:
    if requested_user_id is not None:
        user = db.get(User, requested_user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found.")
        return user

    existing = db.execute(select(User).order_by(User.id.asc())).scalars().first()
    if existing:
        return existing

    mock = User(external_id="mock-local-user", full_name="Local Demo User")
    db.add(mock)
    db.commit()
    db.refresh(mock)
    return mock


@app.post("/upload", response_model=UploadResponse)
async def upload_report(
    file: UploadFile = File(...),
    language: str = Form(default="English"),
    user_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    ext = Path(file.filename or "report.pdf").suffix or ".pdf"
    local_name = f"{uuid.uuid4()}{ext}"
    local_path = UPLOAD_DIR / local_name

    try:
        with local_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        user = _get_or_create_user(db, requested_user_id=user_id)

        extracted_json = run_extractor_agent(str(local_path))
        summary = run_summarizer_agent(extracted_json, language=language)

        report = Report(
            user_id=user.id,
            source_filename=file.filename,
            extracted_data=extracted_json,
            report_summary=summary or extracted_json.get("report_summary"),
            tumor_location=extracted_json.get("tumor_location"),
            t_stage=extracted_json.get("t_stage"),
            n_stage=extracted_json.get("n_stage"),
            crm_status=extracted_json.get("crm_status"),
            emvi_status=extracted_json.get("emvi_status"),
            mrtrg_score=extracted_json.get("mrtrg_score"),
            tumor_deposits=extracted_json.get("tumor_deposits"),
        )
        db.add(report)
        db.commit()
        db.refresh(report)

        return UploadResponse(
            report_id=report.id,
            user_id=user.id,
            extracted_json=extracted_json,
            summary=report.report_summary or "",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {exc}") from exc
    finally:
        file.file.close()


@app.get("/reports/{user_id}", response_model=list[ReportItem])
def get_reports(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    reports = (
        db.execute(
            select(Report).where(Report.user_id == user_id).order_by(Report.created_at.desc())
        )
        .scalars()
        .all()
    )

    return [
        ReportItem(
            id=r.id,
            source_filename=r.source_filename,
            report_summary=r.report_summary,
            t_stage=r.t_stage,
            n_stage=r.n_stage,
            crm_status=r.crm_status,
            emvi_status=r.emvi_status,
            mrtrg_score=r.mrtrg_score,
            tumor_deposits=r.tumor_deposits,
            created_at=r.created_at,
        )
        for r in reports
    ]


@app.get("/reports/{report_id}/chat-history", response_model=list[ChatHistoryItem])
def get_report_chat_history(report_id: int, db: Session = Depends(get_db)):
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    rows = (
        db.execute(
            select(ChatHistory)
            .where(ChatHistory.report_id == report_id)
            .order_by(ChatHistory.created_at.asc())
        )
        .scalars()
        .all()
    )
    return [
        ChatHistoryItem(
            id=row.id,
            role=row.role,
            message=row.message,
            created_at=row.created_at,
        )
        for row in rows
    ]


@app.post("/chat/{report_id}", response_model=ChatResponse)
def chat_with_report(report_id: int, payload: ChatRequest, db: Session = Depends(get_db)):
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")

    history_rows = (
        db.execute(
            select(ChatHistory)
            .where(ChatHistory.report_id == report_id)
            .order_by(ChatHistory.created_at.asc())
        )
        .scalars()
        .all()
    )
    chat_history = [{"role": row.role, "content": row.message} for row in history_rows]

    result = run_chat_agent(
        user_message=payload.message,
        extracted_json=report.extracted_data,
        chat_history=chat_history,
        model_choice=payload.model_choice,
    )

    user_turn = ChatHistory(
        report_id=report_id,
        role="user",
        message=payload.message,
        chat_metadata=None,
    )
    assistant_turn = ChatHistory(
        report_id=report_id,
        role="assistant",
        message=result["answer"],
        chat_metadata=result.get("metadata", {}),
    )
    db.add(user_turn)
    db.add(assistant_turn)
    db.commit()

    return ChatResponse(
        report_id=report_id,
        answer=result["answer"],
        metadata=result.get("metadata", {}),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "env": os.getenv("ENV", "local")}
