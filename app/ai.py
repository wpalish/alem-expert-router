"""AI-слой: извлечение требуемых компетенций из свободного текста заявки.

Координатор пишет заявку обычными словами («нужен бот по базе знаний на
Python») — система сама превращает это в набор компетенций с уровнями.

Два режима:
  • LLM (Groq / OpenAI-совместимый) — если задан GROQ_API_KEY/OPENAI_API_KEY;
  • детерминированный fallback по таксономии — работает всегда, без ключей.

Функция никогда не падает: при любой ошибке возвращает результат fallback.
"""
from __future__ import annotations

import json
import os
import re

import httpx

# канонический навык -> триггеры (рус/англ, в нижнем регистре)
TAXONOMY: dict[str, list[str]] = {
    "python": ["python", "питон", "пайтон"],
    "fastapi": ["fastapi", "фастапи"],
    "django": ["django", "джанго"],
    "react": ["react", "реакт"],
    "typescript": ["typescript", "тайпскрипт"],
    "sql": ["sql", "postgres", "база данных", "бд", "запрос"],
    "ml": ["ml", "machine learning", "машинн", "модел", "обучени"],
    "data": ["data engineer", "данны", "аналитик", "etl", "пайплайн"],
    "nlp": ["nlp", "llm", "чат-бот", "чатбот", "распознавани", "текст", "язык"],
    "devops": ["devops", "ci/cd", "деплой", "kubernetes", "k8s"],
    "docker": ["docker", "докер", "контейнер"],
    "flutter": ["flutter", "флаттер"],
    "ui": ["ui", "ux", "интерфейс", "вёрстк", "верстк", "frontend", "фронт"],
}

_STRONG = re.compile(r"увере|эксперт|senior|глубок|сильн|advanced|продвинут", re.I)
_BASIC = re.compile(r"базов|junior|основ|начальн|джун", re.I)

# отрицание перед триггером: «не нужен python», «без machine learning» и т.п.
_NEG = re.compile(r"\b(не|без|нет|никак\w*|исключ\w*)\b", re.I)


def _level_for(text: str) -> int:
    if _STRONG.search(text):
        return 4
    if _BASIC.search(text):
        return 2
    return 3


def _is_negated(low: str, pos: int) -> bool:
    """Есть ли отрицание в окне ~25 символов перед совпадением."""
    window = low[max(0, pos - 25):pos]
    return bool(_NEG.search(window))


def extract_keywords(text: str) -> dict[str, int]:
    """Детерминированное извлечение по таксономии (fallback, без сети).

    Учитывает отрицание: «бот без ML» не добавит навык `ml`.
    """
    low = text.lower()
    level = _level_for(low)
    found: dict[str, int] = {}
    for skill, triggers in TAXONOMY.items():
        for t in triggers:
            pos = low.find(t)
            if pos != -1 and not _is_negated(low, pos):
                found[skill] = level
                break
    return found


async def _llm_extract(text: str) -> dict[str, int] | None:
    """Извлечение через LLM (Groq). None — если ключа нет или ошибка.

    Асинхронный httpx, чтобы НЕ блокировать event loop FastAPI на время
    сетевого запроса (раньше был sync-вызов внутри обработчика).
    """
    key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    allowed = ", ".join(TAXONOMY)
    prompt = (
        "Ты помогаешь распределять задачи. Извлеки требуемые технические "
        "компетенции из описания. Верни ТОЛЬКО JSON-объект вида "
        '{"навык": уровень}, где навык строго из списка: '
        f"[{allowed}], уровень — целое 1..5. Учитывай отрицания (если навык "
        "явно НЕ нужен — не включай). Без пояснений.\n\nОписание:\n{text}"
    ).replace("{text}", text)
    try:
        async with httpx.AsyncClient(timeout=20) as cli:
            r = await cli.post(
                base,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "temperature": 0,
                      "messages": [{"role": "user", "content": prompt}]},
            )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.S)
        raw = json.loads(match.group(0) if match else content)
        out: dict[str, int] = {}
        for skill, lvl in raw.items():
            s = str(skill).lower().strip()
            if s in TAXONOMY:
                out[s] = max(1, min(5, int(lvl)))
        return out or None
    except Exception:
        return None


async def extract_skills(text: str) -> tuple[dict[str, int], str]:
    """Извлечь компетенции из текста. Возвращает (навыки, метод)."""
    if not text.strip():
        return {}, "empty"
    llm = await _llm_extract(text)
    if llm:
        return llm, "llm"
    return extract_keywords(text), "keywords"
