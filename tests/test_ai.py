"""Тесты AI-извлечения компетенций (детерминированный fallback)."""
from __future__ import annotations

import asyncio

from app.ai import extract_keywords, extract_skills


def test_extracts_positive_skills() -> None:
    out = extract_keywords("Нужен бот на Python с ML")
    assert "python" in out and "ml" in out


def test_negation_excludes_skill() -> None:
    out = extract_keywords("Сайт на Python, без машинного обучения")
    assert "python" in out
    assert "ml" not in out  # отрицание перед триггером


def test_extract_skills_empty() -> None:
    skills, method = asyncio.run(extract_skills("   "))
    assert skills == {} and method == "empty"


def test_extract_skills_uses_keywords_without_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    skills, method = asyncio.run(extract_skills("портал на React и TypeScript"))
    assert method == "keywords"
    assert "react" in skills and "typescript" in skills
