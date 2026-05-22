from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    external_id: Mapped[str | None] = mapped_column(
        String(128), unique=True, index=True, nullable=True
    )
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    reports: Mapped[list["Report"]] = relationship(
        "Report", back_populates="user", cascade="all, delete-orphan"
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Canonical extraction payload from Agent 1. JSON type maps cleanly to PostgreSQL JSONB.
    extracted_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Denormalized fields for frequent filtering/query use.
    report_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tumor_location: Mapped[str | None] = mapped_column(Text, nullable=True)
    t_stage: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    n_stage: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    crm_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    emvi_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    mrtrg_score: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tumor_deposits: Mapped[str | None] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User", back_populates="reports")
    chat_history: Mapped[list["ChatHistory"]] = relationship(
        "ChatHistory", back_populates="report", cascade="all, delete-orphan"
    )


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), index=True, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant
    message: Mapped[str] = mapped_column(Text, nullable=False)
    chat_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    report: Mapped["Report"] = relationship("Report", back_populates="chat_history")
