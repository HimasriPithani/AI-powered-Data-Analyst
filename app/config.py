"""
Centralized configuration for the AI Data Analyst app.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    max_agent_turns: int = int(os.getenv("MAX_AGENT_TURNS", "6"))
    max_rows_preview: int = int(os.getenv("MAX_ROWS_PREVIEW", "200"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "logs/app.log")
    enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    reasoning_effort: str = "low" 


settings = Settings()