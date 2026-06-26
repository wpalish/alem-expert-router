"""AI-слой: извлечение требуемых компетенций из свободного текста заявки.

Координатор пишет заявку обычными словами («нужен бот по базе знаний на
Python») -- система сама превращает это в набор компетенций с уровнями.

Два режима:
  - LLM (Groq / OpenAI-совместимый) -- если задан GROQ_API_KEY/OPENAI_API_KEY;
  - детерминированный fallback по таксономии -- работает всегда, без ключей.

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

# --- П.9 FIX: негативные паттерны (не извлекаем навык из отрицания) ---
_NEGATION = re.compile(
    r"без\s+(?:машинн[огое]{0,3}|ml|python|питон[ау]?)"
    r"|не\s+(?:нуж[ено]|требу[еется]|использ[оватьуям]|нуж[ено])"
    r"|нет\s+(?:необходимости|потребности|надобности)",
    re.I,
)


def _level_for(text: str, skill: str) -> int:
    """Определяет уровень владения навыком по контексту вокруг упоминания."""
    # Ищем контекст: 30 символов до и после первого упоминания навыка/триггера
    low = text.lower()
    triggers = TAXONOMY.get(skill, [])
    for t in triggers:
        idx = low.find(t)
        if idx >= 0:
            # Берём окно ±50 символов для контекста
            start = max(0, idx - 50)
            end = min(len(low), idx + len(t) + 50)
            context = low[start:end]
            if _STRONG.search(context):
                return 4
            if _BASIC.search(context):
                return 2
            break
    return 3


def extract_keywords(text: str) -> dict[str, int]:
    """Детерминированное извлечение по таксономии (fallback, без сети)."""
    low = text.lower()

    # Проверяем негативные паттерны -- если весь текст содержит отрицание,
    # не извлекаем ничего
    if _NEGATION.search(low):
        # Проверяем точнее: негатив может относиться к конкретному навыку
        pass  # продолжаем, негатив проверяется на уровне конкретного триггера

    found: dict[str, int] = {}
    for skill, triggers in TAXONOMY.items():
        if any(t in low for t in triggers):
            trigger_used = next(t for t in triggers if t in low)
            idx = low.find(trigger_used)

            # Отрицание учитываем только если оно стоит НЕПОСРЕДСТВЕННО перед
            # триггером (в пределах ~20 символов и заканчивается у него), иначе
            # «не нужен react» в конце строки ошибочно глушил бы python в начале.
            pre = low[max(0, idx - 22):idx]
            if re.search(r"(?:без|не\s*нуж\w*|не\s*требу\w*|не\s*использ\w*|нет)\s*$", pre):
                continue

            found[skill] = _level_for(text, skill)
    return found


async def _llm_extract(text: str) -> dict[str, int] | None:
    """Извлечение через LLM (Groq). None -- если ключа нет или ошибка."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    base = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    allowed = ", ".join(TAXONOMY)
    # Ограничиваем длину текста -- защита от огромных промптов
    safe_text = text[:2000]
    prompt = (
        "Ты помогаешь распределять задачи. Извлеки требуемые технические "
        "компетенции из описания. Верни ТОЛЬКО JSON-объект вида "
        '{"навык": уровень}, где навык строго из списка: '
        f"[{allowed}], уровень -- целое 1..5. Без пояснений.\n\nОписание:\n{safe_text}"
    )
    try:
        # --- П.10 FIX: async httpx, не блокируем event loop ---
        async with httpx.AsyncClient() as client:
            r = await client.post(
                base,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=20,
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


def extract_skills(text: str) -> tuple[dict[str, int], str]:
    """Извлечь компетенции из текста. Возвращает (навыки, метод)."""
    if not text.strip():
        return {}, "empty"
    # Синхронная обёртка для совместимости
    # В async-контексте используй extract_skills_async
    return extract_keywords(text), "keywords"


async def extract_skills_async(text: str) -> tuple[dict[str, int], str]:
    """Асинхронная версия с попыткой LLM-извлечения."""
    if not text.strip():
        return {}, "empty"
    llm = await _llm_extract(text)
    if llm:
        return llm, "llm"
    return extract_keywords(text), "keywords"