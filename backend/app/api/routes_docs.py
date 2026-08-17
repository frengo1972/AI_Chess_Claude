"""Serves the project documentation so the web UI can render it in-app."""

from __future__ import annotations

import re
from typing import Dict, List

from fastapi import APIRouter, HTTPException

from app.config import DOCS_DIR

router = APIRouter(prefix="/api/docs", tags=["docs"])

_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_SLUG = re.compile(r"^[a-z0-9][a-z0-9\-]*$")


def _documents() -> List[Dict[str, str]]:
    if not DOCS_DIR.exists():
        return []
    entries: List[Dict[str, str]] = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = _HEADING.search(text)
        title = match.group(1).strip() if match else path.stem
        summary = _first_paragraph(text)
        entries.append(
            {
                "slug": path.stem,
                "title": title,
                "summary": summary,
                "words": str(len(text.split())),
            }
        )
    return entries


def _first_paragraph(text: str) -> str:
    body = _HEADING.sub("", text, count=1).strip()
    for block in body.split("\n\n"):
        cleaned = block.strip()
        if cleaned and not cleaned.startswith(("#", "|", "```", "<!--")):
            return " ".join(cleaned.split())[:280]
    return ""


@router.get("")
async def index() -> Dict:
    return {"documents": _documents()}


@router.get("/{slug}")
async def document(slug: str) -> Dict:
    if not _SLUG.match(slug):
        raise HTTPException(status_code=400, detail="invalid document id")
    path = (DOCS_DIR / f"{slug}.md").resolve()
    if not path.is_file() or DOCS_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail=f"unknown document {slug!r}")
    text = path.read_text(encoding="utf-8")
    match = _HEADING.search(text)
    return {
        "slug": slug,
        "title": match.group(1).strip() if match else slug,
        "markdown": text,
    }
