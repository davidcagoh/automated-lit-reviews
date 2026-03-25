from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


class Settings:
    supabase_url: str = ""
    supabase_key: str = ""
    s2_api_key: str | None = None
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""

    def __init__(self) -> None:
        self.supabase_url = _require("SUPABASE_URL")
        # Accept service key or fall back to publishable key (no RLS)
        supabase_key = (
            os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")
        )
        if not supabase_key:
            raise RuntimeError("Missing SUPABASE_SERVICE_KEY or NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY")
        self.supabase_key = supabase_key
        self.s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        # LLM key: prefer Anthropic, fall back to GROQ
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.groq_api_key = os.getenv("GROQ_API_KEY") or ""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or ""


settings = Settings()
