#!/usr/bin/env python3
"""Fatwa Search — Multi-scholar semantic retrieval and AI analysis over trusted Islamic fatwas.

Upgraded to support multiple scholars:
- Sheikh Abdulaziz ibn Baz
- Sheikh Muhammad ibn Saalih al-Uthaymeen
- Sheikh Saleh Al-Fawzan
- Al-Lajnah Ad-Da'imah lil-Ifta / Council of Senior Scholars
and other trusted Saudi Sunni/Salafi authorities.

Two Search Modes:
1. X answers per scholar (independent search per dataset)
2. X answers ranked overall (combined search across selected scholars)

AI Features:
- Individual grounded verdict per fatwa (Gemini, OpenRouter, or JEV).
- Consensus Classification: do the scholars agree or disagree? (Gemini and JEV).
- Comparative Summary: synthesis of scholarly viewpoints (Gemini).
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any
import webbrowser

import gradio as gr
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent


def load_env_files() -> None:
    for env_path in [APP_DIR / ".env", Path(".env")]:
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


load_env_files()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_REPO_ID = os.environ.get("FATWA_DATA_REPO", "ZenoAbderrahim/binbaz_fatwas_rag")
MULTI_SCHOLAR_DATA_FILES = [
    "multi_scholar_clean.csv",
    "qwen3_06b_multi_scholar_embeddings.pt",
    "multi_scholar_metadata.json",
]
BINBAZ_DATA_FILES = [
    "binbaz_clean.csv",
    "qwen3_06b_title_answer_embeddings.pt",
    "embedding_metadata.json",
]
DATA_FILES = MULTI_SCHOLAR_DATA_FILES

# Scholars (exact values of the `source` column) that are searchable/displayed.
# Smaller sources are excluded because their corpora are too small:
# Abdullah al-Manee (233 fatwas), Abdulaziz Aal ash-Sheikh (135), Saleh bin Humaid (48).
# Re-enable any of them by adding their `source` name here. Filtering happens at
# load time only; data files and embeddings on disk are untouched.
INCLUDED_SCHOLARS = ["Ibn Baz", "Ibn Uthaymeen", "Al-Lajnah Ad-Daimah", "Saleh Al-Fawzan"]

EMBEDDER_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
RERANKER_MODEL_NAME = "ALJIACHI/Mizan-Rerank-v3-Turbo"

EMBEDDER_MAX_SEQ_LENGTH = 1024
RERANKER_MAX_LENGTH = 2048
RERANKER_BATCH_SIZE = 16

# ModernBERT compatibility on older GPUs (Pascal capability < 7.0 such as GTX 1050/1060)
if hasattr(torch, "_dynamo"):
    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 7:
            torch._dynamo.config.disable = True
        else:
            torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

DEFAULT_RETRIEVE_K = 20
MAX_RETRIEVE_K = 200
DEFAULT_RERANK_POOL = 20
DEFAULT_SHOW_N = 5
DEFAULT_X_PER_SCHOLAR = 2
DEFAULT_MAX_CANDIDATE_CHARS = 3000

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

PROVIDER_GEMINI = "Gemini"
PROVIDER_OPENROUTER = "OpenRouter"
PROVIDER_LAYA = "Fine-Tuned Laya (Local / Free)"
PROVIDERS = [PROVIDER_GEMINI, PROVIDER_OPENROUTER]
JUDGE_PROVIDERS = [PROVIDER_LAYA, PROVIDER_GEMINI, PROVIDER_OPENROUTER]

LAYA_MODELS = [
    "ZenoAbderrahim/laya-multilingual-fatwa-relevance",
    "convaiinnovations/laya-multilingual",
]
DEFAULT_LAYA_MODEL = LAYA_MODELS[0]

LAYA_QUESTION_SCHEMA = {
    "relevance": {
        "type": "choice",
        "instructions": (
            "Determine whether the Islamic fatwa directly and specifically answers the user's question.\n"
            "- 'answers-it': The fatwa directly and specifically answers the question asked.\n"
            "- 'does-not-answer': The fatwa does NOT directly answer the question (touches on a related topic or is unrelated)."
        ),
        "criteria": {
            "answers-it": "The fatwa directly answers the specific question asked.",
            "does-not-answer": "The fatwa does not directly answer the question.",
        },
    }
}

_LAYA_AGENTS: dict[tuple[str, str], Any] = {}


def get_laya_agent(model_id: str = DEFAULT_LAYA_MODEL, device: str = "cpu") -> Any:
    key = (model_id, device)
    if key not in _LAYA_AGENTS:
        try:
            import laya
        except ImportError:
            raise RuntimeError(
                "The 'laya' package is required for local judging. Run 'pip install laya' or select Gemini / OpenRouter."
            )
        print(f"Loading Laya Agent '{model_id}' on device '{device}'...")
        _LAYA_AGENTS[key] = laya.load(model_id, device=device)
        print(f"✓ Laya Agent '{model_id}' ready on {device}.")
    return _LAYA_AGENTS[key]

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]
POPULAR_OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3.1-flash-lite",
    "google/gemini-3.5-flash",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-r1",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o3-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-3.7-sonnet",
    "qwen/qwen-2.5-72b-instruct",
    "qwen/qwen-2.5-32b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct:free",
    "typesafe/jev-1.13",
    "~typesafe/jev-latest",
]

_CACHED_OPENROUTER_MODELS: list[str] = []


def get_openrouter_models() -> list[str]:
    global _CACHED_OPENROUTER_MODELS
    if _CACHED_OPENROUTER_MODELS:
        return _CACHED_OPENROUTER_MODELS
    try:
        resp = requests.get("https://openrouter.ai/api/v1/models", timeout=3.5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            all_ids = [m["id"] for m in data if "id" in m]
            seen = set()
            ordered = []
            for m in POPULAR_OPENROUTER_MODELS:
                if m not in seen:
                    seen.add(m)
                    ordered.append(m)
            for m in sorted(all_ids):
                if m not in seen:
                    seen.add(m)
                    ordered.append(m)
            _CACHED_OPENROUTER_MODELS = ordered
            return _CACHED_OPENROUTER_MODELS
    except Exception:
        pass
    _CACHED_OPENROUTER_MODELS = list(POPULAR_OPENROUTER_MODELS)
    return _CACHED_OPENROUTER_MODELS


OPENROUTER_CHAT_MODELS = POPULAR_OPENROUTER_MODELS
JEV_MODELS = ["typesafe/jev-1.13", "~typesafe/jev-latest"]

# Modes
SEARCH_MODE_PER_SCHOLAR = "per_scholar"
SEARCH_MODE_OVERALL = "overall"
SEARCH_MODES = [
    ("لكل عالم", SEARCH_MODE_PER_SCHOLAR),
    ("الأفضل إجمالاً", SEARCH_MODE_OVERALL),
]
SEARCH_MODES_EN = [
    ("Per scholar", SEARCH_MODE_PER_SCHOLAR),
    ("Best overall", SEARCH_MODE_OVERALL),
]

# Verdicts for individual fatwas (binary relevance: answers-it or does-not-answer)
VERDICTS = ["answers-it", "does-not-answer"]
VERDICT_LABELS = {
    "answers-it": "Answers it",
    "does-not-answer": "Does not answer",
    "unknown": "Unknown",
}
VERDICT_LABELS_AR = {
    "answers-it": "تجيب على السؤال",
    "does-not-answer": "لا تجيب",
    "unknown": "غير محدد",
}
VERDICT_ICONS = {
    "answers-it": "✓",
    "does-not-answer": "✕",
    "unknown": "?",
}

JEV_CRITERIA = {
    "answers-it": "The fatwa directly answers the user's question or gives the requested ruling/information.",
    "does-not-answer": "The fatwa does not directly answer the question.",
}

# Verdicts for consensus classification
CONSENSUS_AGREE = "agree"
CONSENSUS_PARTIAL = "partially-agree"
CONSENSUS_DISAGREE = "disagree"
CONSENSUS_VERDICTS = [CONSENSUS_AGREE, CONSENSUS_PARTIAL, CONSENSUS_DISAGREE]

CONSENSUS_LABELS = {
    CONSENSUS_AGREE: "Agreement",
    CONSENSUS_PARTIAL: "Partial Agreement",
    CONSENSUS_DISAGREE: "Divergence",
    "unknown": "Indeterminate",
}
CONSENSUS_LABELS_AR = {
    CONSENSUS_AGREE: "متفقون",
    CONSENSUS_PARTIAL: "اتفاق مع تفصيل",
    CONSENSUS_DISAGREE: "مختلفون",
    "unknown": "غير محدد",
}

CONSENSUS_ICONS = {
    CONSENSUS_AGREE: "✓",
    CONSENSUS_PARTIAL: "≈",
    CONSENSUS_DISAGREE: "✕",
    "unknown": "?",
}

JEV_CONSENSUS_CRITERIA = {
    CONSENSUS_AGREE: "All scholars agree or harmonize on the ruling.",
    CONSENSUS_PARTIAL: "Scholars agree generally but diverge on conditions, details, or specific exceptions.",
    CONSENSUS_DISAGREE: "Scholars hold opposing or conflicting verdicts.",
}

# ---------------------------------------------------------------------------
# Internationalization (Pure Arabic & Pure English — No Dual Language Mixing)
# ---------------------------------------------------------------------------
I18N: dict[str, dict[str, str]] = {
    "ar": {
        "title": "محرك البحث الفقهي — فتاوى كبار العلماء",
        "tagline_fmt": "{n:,} فتوى مفهرسة عبر {scholars} من كبار العلماء · بحث دلالي + إعادة ترتيب بميزان v3 Turbo + فحص الصلة بلَيا",
        "theme_light": "نهار",
        "theme_dark": "ليل",
        "lang_btn": "English",
        "settings_btn": "الإعدادات",
        "placeholder": "اكتب سؤالك هنا…",
        "search_mode_label": "الترتيب",
        "mode_per_scholar": "لكل عالم",
        "mode_overall": "الأفضل إجمالاً",
        "x_per_scholar_label": "النتائج لكل عالم",
        "show_n_label": "عدد النتائج",
        "scholars_label": "العلماء",
        "tab_results": "النتائج",
        "tab_pipeline": "المسار الزمني",
        "tab_debug": "التشخيص",
        "empty_query": "",
        "empty_results_title": "جاهز للبحث",
        "empty_results_desc": "اكتب سؤالك الفقهي باللغة العربية أو بأي لغة أخرى للبحث في فتاوى كبار العلماء.",
        "empty_failed": '<div class="fq-empty-sm">لم يعثر البحث على نتائج مطابقة.</div>',
        "loading_query": '<div class="fq-loading"><div class="fq-spin"></div><span>جاري البحث وتحليل الفتاوى الفقهية…</span></div>',
        "loading_results_title": "جاري الاسترجاع والتحليل…",
        "sect_direct_answers": "إجابات مباشرة",
        "sect_other_candidates": "نتائج أخرى ذات صلة",
        "sect_all_candidates": "نتائج البحث",
        "best_match": "الأقرب للسؤال",
        "question_label": "السؤال",
        "full_answer": "نص الإجابة كاملاً",
        "copy_answer": "نسخ الإجابة",
        "copied": "تم النسخ!",
        "original_link": "المصدر ↗",
        "verdict_answers": "تجيب على السؤال",
        "verdict_does_not_answer": "لا تجيب",
        "verdict_unknown": "غير محدد",
        "consensus_agree": "متفقون",
        "consensus_partial": "اتفاق مع تفصيل",
        "consensus_disagree": "مختلفون",
        "summary_title": "ملخص ومقارنة أقوال العلماء",
        "modal_up_title": "تأكيد صحة الإجابة",
        "modal_up_desc": "هل هذه الفتوى تجيب مباشرة وبدقة على سؤالك؟ سيتم حفظها كإجابة معتمدة لتدريب نموذج لَيا.",
        "modal_down_title": "الإبلاغ عن عدم دقة الإجابة",
        "modal_down_desc": "هل هذه الفتوى لا تجيب على سؤالك أو صُنفت بالخطأ؟ سيتم حفظ تقريرك لتحسين وتدريب نموذج لَيا.",
        "modal_comment_placeholder": "ملاحظة أو توضيح إضافي (اختياري)...",
        "modal_submit": "إرسال التقييم",
        "modal_cancel": "إلغاء",
        "toast_success": "جزاكم الله خيراً، تم تسجيل تقييمك لتدريب لَيا.",
        "settings_title": "الإعدادات",
        "settings_sub": "تخصيص نماذج الذكاء الاصطناعي وحدود الاسترجاع والتصفية.",
        "filter_title": "تصفية النتائج بحسب الصلة",
        "filter_info": "إظهار الفتاوى التي تطابق الأحكام المحددة فقط.",
        "reset_btn": "استعادة الإعدادات الافتراضية",
    },
    "en": {
        "title": "Fatwa Search — Multi-Scholar RAG",
        "tagline_fmt": "{n:,} fatwas indexed across {scholars} scholars · semantic retrieval + Mizan v3 Turbo rerank + Laya AI verification",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "lang_btn": "العربية",
        "settings_btn": "Settings",
        "placeholder": "Type your question here…",
        "search_mode_label": "Ranking",
        "mode_per_scholar": "Per scholar",
        "mode_overall": "Best overall",
        "x_per_scholar_label": "Answers per scholar",
        "show_n_label": "Results",
        "scholars_label": "Scholars",
        "tab_results": "Results",
        "tab_pipeline": "Pipeline",
        "tab_debug": "Debug",
        "empty_query": "",
        "empty_results_title": "Ready to search",
        "empty_results_desc": "Enter a question in Arabic or any language. Multiple scholar datasets are indexed and ready to compare.",
        "empty_failed": '<div class="fq-empty-sm">Search produced no matching results.</div>',
        "loading_query": '<div class="fq-loading"><div class="fq-spin"></div><span>Searching and synthesizing across scholarly fatwas…</span></div>',
        "loading_results_title": "Retrieving & analyzing…",
        "sect_direct_answers": "Direct answers",
        "sect_other_candidates": "Other related results",
        "sect_all_candidates": "Search results",
        "best_match": "Best match",
        "question_label": "Question",
        "full_answer": "Full answer",
        "copy_answer": "Copy answer",
        "copied": "Copied!",
        "original_link": "Original ↗",
        "verdict_answers": "Answers it",
        "verdict_does_not_answer": "Does not answer",
        "verdict_unknown": "Unknown",
        "consensus_agree": "Agreement",
        "consensus_partial": "Partial Agreement",
        "consensus_disagree": "Divergence",
        "summary_title": "Comparative Summary of Scholarly Opinions",
        "modal_up_title": "Confirm Direct Answer",
        "modal_up_desc": "Does this fatwa directly and accurately answer your question? It will be saved as a verified answer to train Laya.",
        "modal_down_title": "Report Inaccuracy",
        "modal_down_desc": "Does this fatwa fail to answer your question or was it misclassified? Your report will help improve Laya.",
        "modal_comment_placeholder": "Optional comment or explanation...",
        "modal_submit": "Submit Feedback",
        "modal_cancel": "Cancel",
        "toast_success": "Thanks. Your feedback was saved for retraining Laya.",
        "settings_title": "Settings",
        "settings_sub": "Configure AI features, models, retrieval pools and filters.",
        "filter_title": "Filter Verdicts",
        "filter_info": "Show only fatwas matching the selected verdicts.",
        "reset_btn": "Reset to defaults",
    },
}


def models_for(task: str, provider: str) -> list[str]:
    if provider == PROVIDER_LAYA:
        return list(LAYA_MODELS)
    if provider == PROVIDER_GEMINI:
        return list(GEMINI_MODELS)
    return get_openrouter_models()


def default_model_for(task: str, provider: str) -> str:
    if provider == PROVIDER_LAYA:
        return DEFAULT_LAYA_MODEL
    if provider == PROVIDER_GEMINI:
        return GEMINI_MODELS[0]
    return "google/gemini-2.5-flash"


def is_jev_model(model: str) -> bool:
    m = (model or "").lower()
    return "jev" in m and ("typesafe" in m or m.startswith("~"))


# ---------------------------------------------------------------------------
# Schemas & Data Structures
# ---------------------------------------------------------------------------


class QueryRewrite(BaseModel):
    language: str = Field(description="Detected input language, e.g. Arabic, English, French")
    improved_query: str = Field(description="A clear Arabic search query preserving the user's intent")
    changed: bool = Field(description="Whether translation or reformulation materially changed the query")


class FatwaJudgementModel(BaseModel):
    candidate_id: int
    verdict: str = Field(description="Exactly one of: answers-it, related-only, unrelated")
    confidence: str = Field(description="low, medium, or high")
    answer: str = Field(description="A concise Arabic answer grounded strictly in this fatwa")
    explanation: str = Field(description="Brief Arabic explanation of the classification")


class JudgeResponse(BaseModel):
    judgements: list[FatwaJudgementModel]


class ConsensusClassificationModel(BaseModel):
    consensus: str = Field(description="One of: 'agree', 'partially-agree', 'disagree'")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(description="Concise Arabic explanation analyzing consensus or divergence")


@dataclass
class FatwaJudgement:
    verdict: str
    confidence: str | float | None = None
    answer: str = ""
    explanation: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ConsensusVerdict:
    consensus: str
    confidence: float | None = None
    explanation: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class Stat:
    label: str
    seconds: float = 0.0
    prompt: int = 0
    completion: int = 0
    total: int = 0
    cost: float = 0.0
    provider: str = ""
    model: str = ""
    note: str = ""


def usage_dict(prompt: Any = 0, completion: Any = 0, total: Any = None, cost: Any = 0.0) -> dict[str, Any]:
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "prompt": prompt,
        "completion": completion,
        "total": int(total) if total is not None else prompt + completion,
        "cost": float(cost or 0.0),
    }


# ---------------------------------------------------------------------------
# Human-in-the-Loop Feedback & Active Learning for Laya
# ---------------------------------------------------------------------------
FEEDBACK_FILE_JSONL = "laya_user_feedback.jsonl"
FEEDBACK_FILE_CSV = "laya_user_feedback.csv"
FEEDBACK_FINETUNE_JSONL = "laya_user_annotations_finetune.jsonl"


def get_feedback_count(data_dir: Path | str) -> int:
    fb_path = Path(data_dir) / FEEDBACK_FILE_JSONL
    if not fb_path.exists():
        return 0
    try:
        with open(fb_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def render_feedback_status(data_dir: Path | str, last_msg: str | None = None) -> str:
    count = get_feedback_count(data_dir)
    msg_html = f'<div class="fq-fb-msg success">{esc(last_msg)}</div>' if last_msg else ""
    return (
        f'<div class="fq-fb-status-bar">'
        f'{msg_html}'
        f'<div class="fq-fb-count">إجمالي العينات التدريبية المسجلة: <b>{count:,}</b> عينة (تُجمع لإعادة تدريب لَيا)</div>'
        f'</div>'
    )


def record_user_feedback(
    data_dir: Path | str,
    search_state: dict[str, Any],
    selected_fid: str | None,
    feedback_type: str,
    user_note: str = "",
) -> bool:
    """Log a thumbs-up/down vote as a labeled example for retraining the judge.

    ``feedback_type`` is the vote: "up" (the fatwa answers the question) or
    "down" (it does not). The legacy codes "TP"/"FN" are accepted as aliases.
    The vote is the ground-truth label; comparing it with the judge's verdict
    gives the outcome (TP / FP / FN / TN) used to measure the judge.
    """
    fb_dir = Path(data_dir)
    fb_dir.mkdir(parents=True, exist_ok=True)

    if not selected_fid or not search_state or not search_state.get("candidates"):
        return False

    try:
        fid = int(selected_fid)
    except (ValueError, TypeError):
        return False

    cand = search_state.get("candidates", {}).get(fid)
    if not cand:
        return False

    vote = {"TP": "up", "FN": "down"}.get(feedback_type, feedback_type)
    if vote not in ("up", "down"):
        return False
    user_label = "answers-it" if vote == "up" else "does-not-answer"
    laya_verdict = cand.get("laya_verdict")
    outcome = None  # unknown when the judge did not score this card
    if laya_verdict in ("answers-it", "does-not-answer"):
        predicted_pos = laya_verdict == "answers-it"
        actual_pos = user_label == "answers-it"
        outcome = {(True, True): "TP", (True, False): "FP", (False, True): "FN", (False, False): "TN"}[
            (predicted_pos, actual_pos)
        ]

    user_query = search_state.get("query", "")
    search_query = search_state.get("search_query", "")

    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": timestamp,
        "query": user_query,
        "search_query": search_query,
        "fatwa_id": fid,
        "scholar": cand.get("source", ""),
        "title": cand.get("title", ""),
        "answer_snippet": str(cand.get("answer", ""))[:500],
        "vote": vote,  # "up" / "down" as clicked by the user
        "user_label": user_label,  # ground truth from the vote
        "laya_verdict": laya_verdict,  # judge prediction at the time of the vote
        "outcome": outcome,  # TP / FP / FN / TN of the judge vs the user label
        "laya_prob": cand.get("laya_prob"),
        "reranker_score": cand.get("reranker_score"),
        "user_note": (user_note or "").strip(),
    }

    # 1. JSONL log
    with open(fb_dir / FEEDBACK_FILE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 2. CSV log
    csv_path = fb_dir / FEEDBACK_FILE_CSV
    csv_exists = csv_path.exists()
    pd.DataFrame([entry]).to_csv(csv_path, mode="a", header=not csv_exists, index=False, encoding="utf-8")

    # 3. Laya retraining format: the user's label is the target
    target_choice = user_label
    laya_sample = {
        "state": {
            "user_query": user_query,
            "scholar": cand.get("source", ""),
            "fatwa_title": cand.get("title", ""),
            "fatwa_answer": str(cand.get("answer", ""))[:3000],
        },
        "questions": {
            "relevance": {
                "type": "choice",
                "choice": target_choice,
            }
        },
        "metadata": {
            "vote": vote,
            "laya_verdict": laya_verdict,
            "outcome": outcome,
            "user_note": (user_note or "").strip(),
            "timestamp": timestamp,
        }
    }
    with open(fb_dir / FEEDBACK_FINETUNE_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(laya_sample, ensure_ascii=False) + "\n")

    return True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def parse_categories(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [x.strip() for x in re.split(r"[,;]", text) if x.strip()]


def load_project_assets(data_dir: Path):
    """Loads either multi-scholar corpus assets, or falls back to Bin Baz base corpus."""
    multi_csv = data_dir / "multi_scholar_clean.csv"
    if not multi_csv.exists() and (data_dir / "cleaned" / "multi_scholar_clean.csv").exists():
        multi_csv = data_dir / "cleaned" / "multi_scholar_clean.csv"

    multi_emb = data_dir / "qwen3_06b_multi_scholar_embeddings.pt"
    if not multi_emb.exists() and (data_dir / "cleaned" / "qwen3_06b_multi_scholar_embeddings.pt").exists():
        multi_emb = data_dir / "cleaned" / "qwen3_06b_multi_scholar_embeddings.pt"

    multi_meta = data_dir / "multi_scholar_metadata.json"

    binbaz_csv = data_dir / "binbaz_clean.csv"
    binbaz_emb = data_dir / "qwen3_06b_title_answer_embeddings.pt"
    binbaz_meta = data_dir / "embedding_metadata.json"

    if multi_csv.exists() and multi_emb.exists():
        clean_csv = multi_csv
        embeddings_pt = multi_emb
        metadata_json = multi_meta if multi_meta.exists() else None
        print(f"Loading Multi-Scholar corpus from {clean_csv} and embeddings from {embeddings_pt}...")
    elif binbaz_csv.exists() and binbaz_emb.exists():
        clean_csv = binbaz_csv
        embeddings_pt = binbaz_emb
        metadata_json = binbaz_meta if binbaz_meta.exists() else None
        print(f"Loading Bin Baz corpus from {clean_csv} and embeddings from {embeddings_pt}...")
    else:
        missing = [p for p in (binbaz_csv, binbaz_emb) if not p.exists()]
        raise FileNotFoundError(
            f"Missing required data assets in {data_dir}. Missing:\n" + "\n".join(f"- {p}" for p in missing)
        )

    df = pd.read_csv(clean_csv)
    meta = {}
    if metadata_json and metadata_json.exists():
        with open(metadata_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

    embeddings = torch.load(embeddings_pt, map_location="cpu", weights_only=True)
    if isinstance(embeddings, dict):
        for key in ("embeddings", "tensor", "data"):
            if key in embeddings:
                embeddings = embeddings[key]
                break
    if not torch.is_tensor(embeddings):
        embeddings = torch.as_tensor(embeddings)

    embeddings = embeddings.float().contiguous()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected [N, D] embeddings, got shape {tuple(embeddings.shape)}")
    if embeddings.shape[0] != len(df):
        if embeddings.shape[0] < len(df):
            print(f"Notice: embeddings count ({embeddings.shape[0]:,}) < CSV rows ({len(df):,}). Slicing DataFrame.")
            df = df.iloc[: embeddings.shape[0]].copy()
        else:
            raise ValueError(f"Embedding row count {embeddings.shape[0]} != CSV row count {len(df)}")
    if embeddings.shape[1] != 1024:
        raise ValueError(f"Expected embedding dimension 1024, got {embeddings.shape[1]}")

    embeddings = F.normalize(embeddings, p=2, dim=1)

    required_cols = {"fatwa_id", "source", "title", "question", "answer"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")

    # Restrict to INCLUDED_SCHOLARS, applying the SAME mask to df rows and embedding rows
    # so positional alignment (df.iloc[i] <-> embeddings[i]) is preserved.
    keep = df["source"].isin(INCLUDED_SCHOLARS).to_numpy()
    if keep.any() and not keep.all():
        n_before = len(df)
        df = df.loc[keep].reset_index(drop=True)
        embeddings = embeddings[torch.as_tensor(keep.copy())].contiguous()
        print(f"Filtered to {INCLUDED_SCHOLARS}: {n_before:,} -> {len(df):,} fatwas.")
    elif not keep.any():
        print(f"Warning: none of INCLUDED_SCHOLARS found in corpus sources {sorted(df['source'].dropna().unique())}; keeping all rows.")

    if "categories" not in df.columns:
        df["categories"] = ""
    if "links" not in df.columns:
        df["links"] = ""
    if "embed_text" not in df.columns:
        df["embed_text"] = df["title"].fillna("") + " — " + df["answer"].fillna("")

    df["categories_list"] = df["categories"].apply(parse_categories)
    return df.reset_index(drop=True), embeddings, meta


# ---------------------------------------------------------------------------
# LLM helpers — OpenRouter
# ---------------------------------------------------------------------------


def get_openrouter_key() -> str:
    key = os.getenv(OPENROUTER_API_KEY_ENV)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_API_KEY_ENV} is not set. Put it in .env or export it."
        )
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in response: {text[:200]}")


def openrouter_chat(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = True,
    timeout: int = 150,
) -> tuple[Any, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if expect_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(OPENROUTER_CHAT_URL, headers=_headers(api_key), json=payload, timeout=timeout)
            if resp.status_code >= 400 and expect_json and "response_format" in payload:
                # Some models on OpenRouter do not support response_format; retry without it and extract JSON from output
                payload_no_rf = dict(payload)
                del payload_no_rf["response_format"]
                resp = requests.post(OPENROUTER_CHAT_URL, headers=_headers(api_key), json=payload_no_rf, timeout=timeout)

            if resp.status_code >= 400:
                raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            u = data.get("usage", {}) or {}
            usage = usage_dict(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("total_tokens"), u.get("cost"))
            if expect_json:
                return _extract_json(content), usage
            return (content or "").strip(), usage
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(1.5)
    raise last_err


def call_jev(api_key: str, model: str, state: dict[str, Any], questions: dict[str, Any], timeout: int = 150):
    payload = {"model": model, "state": state, "questions": questions}
    resp = requests.post(OPENROUTER_DECISIONS_URL, headers=_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code >= 400:
        raise RuntimeError(f"JEV HTTP {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    u = data.get("usage", {}) or {}
    usage = usage_dict(u.get("input_tokens"), u.get("output_tokens"), None, u.get("cost"))
    return data.get("answers", {}) or {}, usage


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "You preprocess a user's question for an Arabic Islamic-fatwa retrieval system. "
    "Detect the input language; if it is not Arabic, translate the intent into natural Modern "
    "Standard Arabic; if it is Arabic, reformulate it into a clear, concise search query while "
    "preserving the exact intent. Do NOT answer the question and do NOT introduce facts, names, "
    "rulings, or constraints that were not present. Respond with a single JSON object using keys "
    '"language", "improved_query", and "changed" (a boolean).'
)


def rewrite_user_prompt(query: str) -> str:
    return f'User question: "{query}"\nRespond ONLY with the JSON object.'


def judge_system_prompt() -> str:
    return (
        "You are a strict, objective Islamic fatwa evaluation judge. You are given a user question "
        "and a list of candidate fatwas with their title, question, and full answer. "
        "For EACH candidate decide: does it actually ANSWER the user's question? "
        "Classify each with verdict: 'answers-it' (directly answers), 'related-only' (on topic but does not answer), "
        "or 'unrelated'. Quote or ground a concise Arabic answer from the fatwa text."
    )


def judge_user_prompt(user_query: str, candidates: pd.DataFrame, max_chars: int) -> str:
    lines = [f"User question: {user_query}\n\nCandidate fatwas:"]
    for idx, row in enumerate(candidates.itertuples(index=False)):
        lines.append(f"\n--- Candidate {idx} (Scholar: {row.source}, fatwa_id: {row.fatwa_id}) ---")
        lines.append(f"Title: {str(row.title)[:max_chars]}")
        lines.append(f"Question: {str(row.question)[:max_chars]}")
        lines.append(f"Answer: {str(row.answer)[:max_chars]}")
    lines.append("\nReturn JSON matching the schema for all candidates.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FatwaApp
# ---------------------------------------------------------------------------


class FatwaApp:
    def __init__(self, settings: "Settings"):
        self.s = settings
        self._gemini_client = None
        self._openrouter_key: str | None = None

        print(f"Data dir: {settings.data_dir}")
        print(f"Embedder device: {settings.device_embedder}")
        print(f"Reranker device: {settings.device_reranker}")
        print(f"Judge device: {settings.device_judge}")

        self.df, self.corpus_embeddings, self.embedding_meta = load_project_assets(settings.data_dir)
        self.available_scholars = sorted(self.df["source"].dropna().unique().tolist())
        print(
            f"Loaded {len(self.df):,} fatwas · embeddings {tuple(self.corpus_embeddings.shape)} · "
            f"Scholars: {self.available_scholars}"
        )

        print(f"Loading embedder ({EMBEDDER_MODEL_NAME}) on {settings.device_embedder}...")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=settings.device_embedder)
        self.embedder.max_seq_length = EMBEDDER_MAX_SEQ_LENGTH

        print(f"Loading reranker ({RERANKER_MODEL_NAME}) on {settings.device_reranker}...")
        # fp16 on CUDA halves the reranker's memory, which lets it share a
        # 3 GB laptop GPU with the Laya judge.
        reranker_kwargs = {"dtype": torch.float16} if settings.device_reranker == "cuda" else {}
        self.reranker = CrossEncoder(
            RERANKER_MODEL_NAME,
            max_length=RERANKER_MAX_LENGTH,
            device=settings.device_reranker,
            trust_remote_code=True,
            model_kwargs=reranker_kwargs,
            # ModernBERT otherwise may torch.compile parts of the model with Triton,
            # which fails on pre-Volta GPUs (e.g. GTX 10xx, compute capability < 7.0).
            config_kwargs={"reference_compile": False},
        )
        print("Models loaded.")

        if settings.warmup:
            self._warmup()

    # -- clients -----------------------------------------------------------

    def ensure_gemini(self, custom_key: str | None = None):
        api_key = (custom_key or "").strip() or os.getenv(GEMINI_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{GEMINI_API_KEY_ENV} is not set. Put it in .env, export it, or enter it in Settings."
            )
        if self._gemini_client is None or getattr(self, "_gemini_key_used", None) != api_key:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            self._gemini_key_used = api_key
        return self._gemini_client

    def ensure_openrouter(self, custom_key: str | None = None) -> str:
        api_key = (custom_key or "").strip() or os.getenv(OPENROUTER_API_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{OPENROUTER_API_KEY_ENV} is not set. Put it in .env, export it, or enter it in Settings."
            )
        self._openrouter_key = api_key
        return self._openrouter_key

    # -- rewrite -----------------------------------------------------------

    def rewrite_query(self, provider: str, model: str, query: str) -> tuple[QueryRewrite, dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            client = self.ensure_gemini()
            response = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=f"{REWRITE_SYSTEM}\n\n{rewrite_user_prompt(query)}",
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": QueryRewrite.model_json_schema(),
                        },
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1.5 * (attempt + 1))

            um = getattr(response, "usage_metadata", None)
            usage = usage_dict(
                getattr(um, "prompt_token_count", 0),
                getattr(um, "candidates_token_count", 0),
                getattr(um, "total_token_count", None),
            )
            return QueryRewrite.model_validate_json(response.text), usage

        key = self.ensure_openrouter()
        data, usage = openrouter_chat(key, model, REWRITE_SYSTEM, rewrite_user_prompt(query))
        if isinstance(data, dict):
            improved = data.get("improved_query") or data.get("improvedQuery") or data.get("query") or query
            lang = data.get("language") or "Arabic"
            changed = bool(data.get("changed", True))
            return QueryRewrite(language=str(lang), improved_query=str(improved), changed=changed), usage
        return QueryRewrite(language="Arabic", improved_query=query, changed=False), usage

    # -- candidate judging -------------------------------------------------

    def judge_candidates(
        self, provider: str, model: str, user_query: str, candidates: pd.DataFrame, max_chars: int
    ) -> tuple[dict[int, FatwaJudgement], dict[str, Any]]:
        if provider == PROVIDER_LAYA:
            return self._judge_laya(model, user_query, candidates, max_chars)
        if provider == PROVIDER_GEMINI:
            return self._judge_gemini(model, user_query, candidates, max_chars)
        key = self.ensure_openrouter()
        if is_jev_model(model):
            return self._judge_jev(key, model, user_query, candidates, max_chars)
        return self._judge_openrouter_chat(key, model, user_query, candidates, max_chars)

    def _judge_laya(
        self, model: str, user_query: str, candidates: pd.DataFrame, max_chars: int
    ) -> tuple[dict[int, FatwaJudgement], dict[str, Any]]:
        judge_device = getattr(getattr(self, "s", None), "device_judge", "cpu")
        agent = get_laya_agent(model, device=judge_device)
        out: dict[int, FatwaJudgement] = {}
        for row in candidates.itertuples(index=False):
            fid = int(row.fatwa_id)
            state = {
                "user_query": str(user_query or "").strip(),
                "scholar": str(row.source).strip(),
                "fatwa_title": str(row.title).strip(),
                "fatwa_answer": str(row.answer)[:max_chars].strip(),
            }
            try:
                res = agent.predict(state, LAYA_QUESTION_SCHEMA)
                ans = res.get("answers", {}).get("relevance", {})
                choice = str(ans.get("choice", "unknown")).strip()
                probs = ans.get("probabilities", {}) or {}
                conf = ans.get("answer_confidence")
                if conf is None and probs and choice in probs:
                    conf = probs[choice]
                elif conf is None:
                    conf = ans.get("confidence")

                p_ans = (
                    probs.get("answers-it", 0.0)
                    if probs
                    else (conf if choice == "answers-it" else (1.0 - conf if conf else 0.0))
                )
                p_ans_pct = f"{p_ans * 100:.1f}%"
                is_direct = choice == "answers-it"
                explanation = (
                    f"Laya neural verdict: {'Direct Answer' if is_direct else 'Does Not Answer'} "
                    f"(P(answers-it) = {p_ans_pct})"
                )
                out[fid] = FatwaJudgement(
                    verdict=choice,
                    confidence=conf,
                    answer="",
                    explanation=explanation,
                    probabilities=probs,
                )
            except Exception as e:
                out[fid] = FatwaJudgement(
                    verdict="unknown",
                    confidence=None,
                    answer="",
                    explanation=f"Laya error: {e}",
                )

        usage = usage_dict(prompt=0, completion=0, total=0, cost=0.0)
        return out, usage

    def _judge_gemini(self, model, user_query, candidates, max_chars):
        client = self.ensure_gemini()
        response = client.models.generate_content(
            model=model,
            contents=f"{judge_system_prompt()}\n\n{judge_user_prompt(user_query, candidates, max_chars)}",
            config={
                "response_mime_type": "application/json",
                "response_schema": JudgeResponse.model_json_schema(),
            },
        )
        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        parsed = JudgeResponse.model_validate_json(response.text)
        out = {
            int(j.candidate_id): FatwaJudgement(
                verdict=j.verdict, confidence=j.confidence, answer=j.answer, explanation=j.explanation
            )
            for j in parsed.judgements
        }
        return out, usage

    def _judge_openrouter_chat(self, key, model, user_query, candidates, max_chars):
        data, usage = openrouter_chat(
            key, model, judge_system_prompt(), judge_user_prompt(user_query, candidates, max_chars)
        )
        judgements_list = []
        if isinstance(data, list):
            judgements_list = data
        elif isinstance(data, dict):
            for k in ("judgements", "judgments", "candidates", "results"):
                if k in data and isinstance(data[k], list):
                    judgements_list = data[k]
                    break
            if not judgements_list and "candidate_id" in data:
                judgements_list = [data]

        out: dict[int, FatwaJudgement] = {}
        for j in judgements_list:
            if not isinstance(j, dict):
                continue
            cid = j.get("candidate_id")
            if cid is None:
                continue
            try:
                cid = int(cid)
            except (ValueError, TypeError):
                continue

            verdict = str(j.get("verdict", "unknown")).strip().lower().replace("_", "-")
            if verdict not in VERDICTS:
                if "answer" in verdict:
                    verdict = "answers-it"
                elif "related" in verdict:
                    verdict = "related-only"
                elif "unrelated" in verdict:
                    verdict = "unrelated"
                else:
                    verdict = "unknown"

            conf = str(j.get("confidence", "medium")).lower()
            ans = str(j.get("answer", "")).strip()
            exp = str(j.get("explanation", "")).strip()
            out[cid] = FatwaJudgement(verdict=verdict, confidence=conf, answer=ans, explanation=exp)

        for row in candidates.itertuples(index=False):
            fid = int(row.fatwa_id)
            if fid not in out:
                out[fid] = FatwaJudgement(verdict="unknown", confidence="low", answer="", explanation="Unclassified")

        return out, usage

    def _judge_jev(self, key, model, user_query, candidates, max_chars):
        state: dict[str, Any] = {"user_query": (user_query or "").strip(), "candidates": {}}
        questions: dict[str, Any] = {}
        ids: list[int] = []
        for idx, row in enumerate(candidates.itertuples(index=False)):
            qid = f"cand_{idx}"
            ids.append(int(row.fatwa_id))
            state["candidates"][qid] = {
                "scholar": str(row.source),
                "title": str(row.title)[:max_chars],
                "question": str(row.question)[:max_chars],
                "answer": str(row.answer)[:max_chars],
            }
            questions[qid] = {
                "type": "choice",
                "instructions": (
                    f"Judge candidate `candidates.{qid}`. Decide whether its fatwa answers the "
                    f"user's question in `user_query`. Pick exactly one option."
                ),
                "criteria": dict(JEV_CRITERIA),
            }

        answers, usage = call_jev(key, model, state, questions)
        out: dict[int, FatwaJudgement] = {}
        for idx, fatwa_id in enumerate(ids):
            answer = answers.get(f"cand_{idx}", {}) or {}
            choice = answer.get("choice")
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities") or {}
            if not isinstance(choice, str) or choice not in VERDICTS:
                choice = choice or "unknown"
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            probabilities = {
                str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))
            }
            out[fatwa_id] = FatwaJudgement(
                verdict=choice, confidence=confidence, probabilities=probabilities
            )
        return out, usage

    # -- AI consensus classification & summary -----------------------------

    def _format_fatwas_for_synthesis(self, candidates: pd.DataFrame, max_chars: int = 1500) -> str:
        blocks = []
        for i, row in enumerate(candidates.itertuples(index=False), 1):
            blocks.append(
                f"[{i}] Scholar: {row.source}\n"
                f"Title: {row.title}\n"
                f"Question: {str(row.question)[:400]}\n"
                f"Answer: {str(row.answer)[:max_chars]}\n"
            )
        return "\n".join(blocks)

    def classify_consensus(
        self, provider: str, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[ConsensusVerdict, dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            return self._consensus_gemini(model, user_query, candidates)
        key = self.ensure_openrouter()
        if is_jev_model(model):
            return self._consensus_jev(key, model, user_query, candidates)
        return self._consensus_openrouter_chat(key, model, user_query, candidates)

    def _consensus_gemini(self, model: str, user_query: str, candidates: pd.DataFrame):
        client = self.ensure_gemini()
        prompt = (
            f"User Question: {user_query}\n\n"
            f"Scholarly Fatwas:\n{self._format_fatwas_for_synthesis(candidates)}\n\n"
            "Task: Compare the rulings and answers given by the scholars to this question. "
            "Classify their degree of consensus into one of three values:\n"
            "- 'agree': All scholars agree on the core ruling.\n"
            "- 'partially-agree': Scholars agree on the main principle but differ in conditions, details, or exceptions.\n"
            "- 'disagree': Scholars hold conflicting rulings (e.g. permissible vs prohibited, obligatory vs recommended).\n"
            "Provide a clear, respectful Arabic explanation of the agreement or divergence."
        )
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ConsensusClassificationModel.model_json_schema(),
                    },
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))

        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        parsed = ConsensusClassificationModel.model_validate_json(response.text)
        choice = parsed.consensus if parsed.consensus in CONSENSUS_VERDICTS else "unknown"
        return ConsensusVerdict(
            consensus=choice,
            confidence=parsed.confidence,
            explanation=parsed.explanation,
        ), usage

    def _consensus_jev(self, key: str, model: str, user_query: str, candidates: pd.DataFrame):
        state = {
            "user_query": (user_query or "").strip(),
            "scholars": list(candidates["source"].unique()),
            "fatwa_answers": [
                {"scholar": str(row.source), "title": str(row.title), "answer": str(row.answer)[:1000]}
                for row in candidates.itertuples(index=False)
            ],
        }
        questions = {
            "scholarly_consensus": {
                "type": "choice",
                "instructions": (
                    "Based on the provided scholarly answers, analyze whether the scholars agree on the ruling for the user's question. "
                    "Select 'agree' if their conclusions and rulings are identical or compatible. "
                    "Select 'partially-agree' if they agree on the general principle but differ in conditions, details, or branches. "
                    "Select 'disagree' if they express conflicting rulings (e.g. permissible vs prohibited)."
                ),
                "criteria": dict(JEV_CONSENSUS_CRITERIA),
            }
        }
        answers, usage = call_jev(key, model, state, questions)
        ans = answers.get("scholarly_consensus", {}) or {}
        choice = ans.get("choice") or "unknown"
        confidence = ans.get("confidence")
        probabilities = ans.get("probabilities") or {}
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None

        labels_ar = {
            CONSENSUS_AGREE: "اتفاق بين العلماء في أصل الحكم",
            CONSENSUS_PARTIAL: "اتفاق على أصل المسألة مع تفصيل وتنوع في الفروع أو الشروط",
            CONSENSUS_DISAGREE: "اختلاف في الحكم بين العلماء المذكورين",
        }
        explanation = labels_ar.get(choice, f"تصنيف التوافق بواسطة JEV: {choice}")
        return ConsensusVerdict(
            consensus=choice if choice in CONSENSUS_VERDICTS else "unknown",
            confidence=confidence,
            explanation=explanation,
            probabilities={str(k): float(v) for k, v in probabilities.items() if isinstance(v, (int, float))},
        ), usage

    def _consensus_openrouter_chat(self, key: str, model: str, user_query: str, candidates: pd.DataFrame):
        sys_prompt = (
            "You are an Islamic jurisprudence scholar analysis assistant. "
            "Analyze the retrieved scholarly fatwas and determine their consensus level on the user question. "
            "Return a JSON object with keys:\n"
            "- 'consensus': exactly one of 'agree', 'partially-agree', 'disagree'\n"
            "- 'confidence': float between 0.0 and 1.0\n"
            "- 'explanation': concise Arabic explanation analyzing consensus or divergence."
        )
        user_prompt = f"Question: {user_query}\n\nFatwas:\n{self._format_fatwas_for_synthesis(candidates)}"
        data, usage = openrouter_chat(key, model, sys_prompt, user_prompt)

        choice = "unknown"
        conf = 1.0
        exp = ""
        if isinstance(data, dict):
            raw_c = str(data.get("consensus", "")).strip().lower().replace("_", "-")
            if raw_c in CONSENSUS_VERDICTS:
                choice = raw_c
            elif "part" in raw_c and "agree" in raw_c:
                choice = CONSENSUS_PARTIAL
            elif "agree" in raw_c:
                choice = CONSENSUS_AGREE
            elif "disagree" in raw_c or "diff" in raw_c:
                choice = CONSENSUS_DISAGREE

            try:
                conf = float(data.get("confidence", 1.0))
            except (ValueError, TypeError):
                conf = 1.0
            exp = str(data.get("explanation", "")).strip()

        return ConsensusVerdict(
            consensus=choice,
            confidence=conf,
            explanation=exp,
        ), usage

    def summarize_opinions_gemini(
        self, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[str, dict[str, Any]]:
        client = self.ensure_gemini()
        prompt = (
            f"السؤال: {user_query}\n\n"
            f"فتاوى العلماء المعثور عليها:\n{self._format_fatwas_for_synthesis(candidates)}\n\n"
            "المطلوب: قدم خلاصة فقهية جامعة وموجزة ودقيقة باللغة العربية تبين موقف كل شيخ أو هيئة إفتاء من المسألة. "
            "اذكر أوجه الاتفاق والاختلاف (إن وجدت)، وانسب كل قول إلى قائله بأمانة ودقة واختصار، وبأسلوب علمي رصين."
        )
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))

        um = getattr(response, "usage_metadata", None)
        usage = usage_dict(
            getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0),
            getattr(um, "total_token_count", None),
        )
        return response.text.strip(), usage

    def summarize_opinions_openrouter(
        self, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[str, dict[str, Any]]:
        key = self.ensure_openrouter()
        sys_prompt = (
            "أنت باحث ومساعد متخصص في الفقه الإسلامي المقارن. "
            "مهمتك صياغة خلاصة فقهية موجزة ودقيقة باللغة العربية تبين موقف كل شيخ أو هيئة إفتاء من المسألة المعروضة في الفتاوى المرفقة. "
            "اذكر أوجه الاتفاق والاختلاف (إن وجدت)، وانسب كل قول إلى قائله بأمانة ودقة واختصار، وبأسلوب علمي رصين."
        )
        user_prompt = (
            f"السؤال: {user_query}\n\n"
            f"فتاوى العلماء المعثور عليها:\n{self._format_fatwas_for_synthesis(candidates)}\n\n"
            "المطلوب: قدم الخلاصة الفقهية الجامعة مع نسبة كل حكم إلى صاحبه."
        )
        content, usage = openrouter_chat(key, model, sys_prompt, user_prompt, expect_json=False)
        return content.strip(), usage

    def summarize_opinions(
        self, provider: str, model: str, user_query: str, candidates: pd.DataFrame
    ) -> tuple[str, dict[str, Any]]:
        if provider == PROVIDER_GEMINI:
            return self.summarize_opinions_gemini(model, user_query, candidates)
        return self.summarize_opinions_openrouter(model, user_query, candidates)

    # -- retrieval / reranking --------------------------------------------

    @functools.lru_cache(maxsize=256)
    def _encode_query_cached(self, query: str) -> torch.Tensor:
        vec = self.embedder.encode(
            query,
            convert_to_tensor=True,
            device=self.s.device_embedder,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.detach().cpu().float()

    def retrieve_top_k(
        self, query: str, top_k: int, allowed_scholars: list[str] | None = None
    ) -> pd.DataFrame:
        top_k = max(1, min(int(top_k), MAX_RETRIEVE_K, len(self.df)))
        q_vec = self._encode_query_cached(query)

        if allowed_scholars and len(allowed_scholars) < len(self.available_scholars):
            mask = self.df["source"].isin(allowed_scholars).values
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return pd.DataFrame()
            sub_emb = self.corpus_embeddings[indices]
            scores = torch.mv(sub_emb, q_vec)
            k = min(top_k, len(indices))
            values, top_rel_idx = torch.topk(scores, k=k)
            top_abs_idx = indices[top_rel_idx.numpy()]
            out = self.df.iloc[top_abs_idx].copy().reset_index(drop=True)
            out.insert(0, "retrieval_rank", np.arange(1, len(out) + 1))
            out["embedding_score"] = values.numpy()
            return out
        else:
            scores = torch.mv(self.corpus_embeddings, q_vec)
            values, indices = torch.topk(scores, k=top_k)
            out = self.df.iloc[indices.numpy()].copy().reset_index(drop=True)
            out.insert(0, "retrieval_rank", np.arange(1, len(out) + 1))
            out["embedding_score"] = values.numpy()
            return out

    def retrieve_top_k_per_scholar(
        self, query: str, scholars: list[str], pool_per_scholar: int
    ) -> dict[str, pd.DataFrame]:
        q_vec = self._encode_query_cached(query)
        out = {}
        for s in scholars:
            mask = (self.df["source"] == s).values
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            sub_emb = self.corpus_embeddings[indices]
            scores = torch.mv(sub_emb, q_vec)
            k = max(1, min(int(pool_per_scholar), len(indices)))
            values, top_rel_idx = torch.topk(scores, k=k)
            top_abs_idx = indices[top_rel_idx.numpy()]
            sub_df = self.df.iloc[top_abs_idx].copy().reset_index(drop=True)
            sub_df.insert(0, "retrieval_rank", np.arange(1, len(sub_df) + 1))
            sub_df["embedding_score"] = values.numpy()
            out[s] = sub_df
        return out

    def rerank_candidates(self, query: str, candidates: pd.DataFrame) -> pd.DataFrame:
        if not len(candidates):
            return candidates
        pairs = [[query, text] for text in candidates["embed_text"].astype(str).tolist()]
        scores = np.asarray(self._predict_with_oom_retry(pairs)).reshape(-1)
        reranked = candidates.copy()
        reranked["reranker_score"] = scores
        reranked = reranked.sort_values("reranker_score", ascending=False).reset_index(drop=True)
        reranked.insert(0, "rerank_rank", np.arange(1, len(reranked) + 1))
        return reranked

    def _predict_with_oom_retry(self, pairs: list[list[str]]):
        """Rerank, halving the batch size on CUDA out-of-memory instead of failing the search."""
        batch_size = RERANKER_BATCH_SIZE
        while True:
            try:
                with torch.inference_mode():
                    return self.reranker.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"Reranker hit CUDA OOM; retrying with batch_size={batch_size}")

    def _warmup(self):
        try:
            t0 = time.perf_counter()
            dummy = "ما هي كفارة اليمين؟"
            self.rerank_candidates(dummy, self.retrieve_top_k(dummy, DEFAULT_SHOW_N))
            print(f"Warmup done in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            print(f"Warmup skipped: {type(e).__name__}: {e}")

    # -- main search -------------------------------------------------------

    def run_search(
        self,
        user_query: str,
        search_mode: str,
        selected_scholars: list[str],
        x_per_scholar: int,
        rewrite_enabled: bool,
        rewrite_provider: str,
        rewrite_model: str,
        judge_enabled: bool,
        judge_provider: str,
        judge_model: str,
        consensus_enabled: bool,
        consensus_provider: str,
        consensus_model: str,
        summary_enabled: bool,
        summary_provider: str,
        summary_model: str,
        retrieve_k: int,
        rerank_pool: int,
        show_n: int,
        max_candidate_chars: int,
        min_rerank_score: float | None,
        filter_verdicts: list[str],
        gemini_api_key: str = "",
        openrouter_api_key: str = "",
        lang: str = "ar",
        progress_cb: Any = None,
    ):
        steps: list[Stat] = []
        errors: list[str] = []
        t_start = time.perf_counter()

        if gemini_api_key and gemini_api_key.strip():
            os.environ[GEMINI_API_KEY_ENV] = gemini_api_key.strip()
            self._gemini_client = None
        if openrouter_api_key and openrouter_api_key.strip():
            os.environ[OPENROUTER_API_KEY_ENV] = openrouter_api_key.strip()
            self._openrouter_key = None

        user_query = (user_query or "").strip()
        if not user_query:
            return "Please enter a question.", "", pd.DataFrame(), pd.DataFrame(), "", ""

        # Default scholars if none selected
        if not selected_scholars:
            selected_scholars = list(self.available_scholars)

        # -- rewrite
        rewrite = None
        search_query = user_query
        if progress_cb:
            progress_cb(0.1, "Analyzing question..." if lang == "en" else "جاري تحليل السؤال...")
        if rewrite_enabled:
            t0 = time.perf_counter()
            try:
                rewrite, usage = self.rewrite_query(rewrite_provider, rewrite_model, user_query)
                search_query = rewrite.improved_query.strip() or user_query
                steps.append(
                    Stat("Query rewrite", time.perf_counter() - t0, **usage, provider=rewrite_provider, model=rewrite_model)
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Query rewrite failed ({rewrite_provider} / {rewrite_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Query rewrite", dt, provider=rewrite_provider, model=rewrite_model, note="failed"))
        else:
            steps.append(Stat("Query rewrite", 0.0, note="disabled"))

        # -- retrieval & rerank based on search mode
        retrieved = pd.DataFrame()
        top_n = pd.DataFrame()

        if progress_cb:
            progress_cb(0.25, "Retrieving candidate fatwas..." if lang == "en" else "جاري استرجاع الفتاوى...")

        t0_retrieval = time.perf_counter()
        try:
            if search_mode == SEARCH_MODE_PER_SCHOLAR:
                # MODE 1: X answers per scholar (independent search per dataset)
                pool_s = max(int(x_per_scholar), min(int(rerank_pool), 20))
                per_scholar_cands = self.retrieve_top_k_per_scholar(search_query, selected_scholars, pool_s)
                retrieval_time = time.perf_counter() - t0_retrieval

                if progress_cb:
                    progress_cb(0.5, "Reranking with Mizan v3 Turbo..." if lang == "en" else "جاري إعادة الترتيب بميزان v3 Turbo...")

                t0_rerank = time.perf_counter()
                all_retrieved_list = []
                reranked_parts = []
                for s in selected_scholars:
                    cand_s = per_scholar_cands.get(s)
                    if cand_s is not None and len(cand_s) > 0:
                        all_retrieved_list.append(cand_s)
                        reranked_s = self.rerank_candidates(search_query, cand_s)
                        if min_rerank_score is not None:
                            reranked_s = reranked_s[reranked_s["reranker_score"] >= float(min_rerank_score)].reset_index(drop=True)
                            reranked_s["rerank_rank"] = np.arange(1, len(reranked_s) + 1)
                        reranked_parts.append(reranked_s.head(int(x_per_scholar)))

                rerank_time = time.perf_counter() - t0_rerank
                retrieved = pd.concat(all_retrieved_list, ignore_index=True) if all_retrieved_list else pd.DataFrame()
                if reranked_parts:
                    # Keep each scholar's quota, but order the combined list by
                    # rerank score so "best match" really is the best match.
                    top_n = pd.concat(reranked_parts, ignore_index=True)
                    top_n = top_n.sort_values("reranker_score", ascending=False, kind="stable").reset_index(drop=True)
                    top_n["rerank_rank"] = np.arange(1, len(top_n) + 1)
                else:
                    top_n = pd.DataFrame()

                steps.append(Stat("Embedding retrieval", retrieval_time, note=f"{len(retrieved)} candidates across {len(selected_scholars)} scholars"))
                steps.append(Stat("Mizan v3 Turbo rerank", rerank_time, note=f"{len(top_n)} selected ({x_per_scholar}/scholar)"))

            else:
                # MODE 2: X answers ranked overall
                retrieved = self.retrieve_top_k(search_query, int(retrieve_k), allowed_scholars=selected_scholars)
                retrieval_time = time.perf_counter() - t0_retrieval
                steps.append(Stat("Embedding retrieval", retrieval_time, note=f"{len(retrieved)} candidates"))

                if progress_cb:
                    progress_cb(0.5, "Reranking with Mizan v3 Turbo..." if lang == "en" else "جاري إعادة الترتيب بميزان v3 Turbo...")

                t0_rerank = time.perf_counter()
                pool = min(int(rerank_pool), len(retrieved))
                reranked = self.rerank_candidates(search_query, retrieved.head(pool))
                if min_rerank_score is not None:
                    reranked = reranked[reranked["reranker_score"] >= float(min_rerank_score)].reset_index(drop=True)
                    reranked["rerank_rank"] = np.arange(1, len(reranked) + 1)
                rerank_time = time.perf_counter() - t0_rerank
                steps.append(Stat("Mizan v3 Turbo rerank", rerank_time, note=f"over {pool}"))
                top_n = reranked.head(max(1, int(show_n))).copy()

        except Exception as e:
            err = positional_error(e)
            return self._fatal(user_query, steps, errors + [f"Retrieval/Rerank failed: {e}"], err, lang=lang)

        # -- individual candidate judging
        judge_map: dict[int, FatwaJudgement] = {}
        if judge_enabled and len(top_n):
            if progress_cb:
                progress_cb(0.75, "Evaluating relevance with Laya..." if lang == "en" else "جاري فحص الصلة بنموذج لَيا...")
            t0 = time.perf_counter()
            try:
                judge_map, usage = self.judge_candidates(
                    judge_provider, judge_model, user_query, top_n, int(max_candidate_chars)
                )
                steps.append(
                    Stat("Judge", time.perf_counter() - t0, **usage, provider=judge_provider, model=judge_model, note=f"{len(judge_map)} judged")
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Judge failed ({judge_provider} / {judge_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Judge", dt, provider=judge_provider, model=judge_model, note="failed"))
        else:
            steps.append(Stat("Judge", 0.0, note="disabled" if not judge_enabled else "no candidates"))

        # -- consensus classification (Gemini or JEV)
        consensus_verdict: ConsensusVerdict | None = None
        if consensus_enabled and len(top_n) > 1:
            if progress_cb:
                progress_cb(0.88, "Classifying scholarly consensus..." if lang == "en" else "جاري تحليل التوافق الفقهي...")
            t0 = time.perf_counter()
            try:
                consensus_verdict, usage = self.classify_consensus(
                    consensus_provider, consensus_model, user_query, top_n
                )
                steps.append(
                    Stat("Consensus classification", time.perf_counter() - t0, **usage,
                         provider=consensus_provider, model=consensus_model, note=consensus_verdict.consensus)
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Consensus classification failed ({consensus_provider} / {consensus_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Consensus classification", dt, provider=consensus_provider, model=consensus_model, note="failed"))
        else:
            steps.append(Stat("Consensus classification", 0.0, note="disabled" if not consensus_enabled else "insufficient results"))

        # -- comparative summary
        summary_text: str | None = None
        if summary_enabled and len(top_n) > 0:
            if progress_cb:
                progress_cb(0.94, "Synthesizing comparative summary..." if lang == "en" else "جاري تلخيص أقوال العلماء...")
            t0 = time.perf_counter()
            try:
                summary_text, usage = self.summarize_opinions(summary_provider, summary_model, user_query, top_n)
                steps.append(
                    Stat("Comparative summary", time.perf_counter() - t0, **usage,
                         provider=summary_provider, model=summary_model, note=f"{len(summary_text)} chars")
                )
            except Exception as e:
                dt = time.perf_counter() - t0
                msg = f"Comparative summary failed ({summary_provider} / {summary_model}): {type(e).__name__}: {e}"
                errors.append(msg)
                steps.append(Stat("Comparative summary", dt, provider=summary_provider, model=summary_model, note="failed"))
        else:
            steps.append(Stat("Comparative summary", 0.0, note="disabled" if not summary_enabled else "no candidates"))

        total_seconds = time.perf_counter() - t_start
        steps.append(Stat("Total", total_seconds, note=""))

        # -- verdict filtering
        if judge_enabled and judge_map and filter_verdicts:
            allowed = set(filter_verdicts)
            keep = top_n["fatwa_id"].map(
                lambda fid: (judge_map[int(fid)].verdict in allowed) if int(fid) in judge_map else True
            )
            display_rows = top_n[keep].reset_index(drop=True)
        else:
            display_rows = top_n

        query_info = self._query_panel(user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled, consensus_verdict, lang=lang)
        stats_html = self._stats_html(steps, lang=lang)
        retrieval_view = dataframe_for_display(retrieved)
        rerank_view = dataframe_for_display(top_n, judge_map=judge_map if judge_enabled else None)

        # Consensus Box
        consensus_box_html = render_consensus_box(consensus_verdict, summary_text, lang=lang)

        cards = render_results_cards(
            display_rows,
            judge_map=judge_map if judge_enabled else None,
            judge_enabled=judge_enabled,
            consensus_box_html=consensus_box_html,
            lang=lang,
        )

        if progress_cb:
            progress_cb(1.0, "Complete!" if lang == "en" else "اكتمل البحث!")

        error_text = "\n\n".join(errors)
        if errors:
            gr.Warning(error_modal_html(errors, user_query, steps), duration=None, title="Error — copy and send to the AI")

        return query_info, stats_html, retrieval_view, rerank_view, cards, error_text, display_rows, judge_map, search_query

    def _fatal(self, user_query, steps, errors, err, lang="ar"):
        steps.append(Stat("Total", sum(s.seconds for s in steps if s.label != "Total"), note="aborted"))
        tb = traceback.format_exc() if err is None else err
        gr.Warning(error_modal_html(errors, user_query, steps, tb), duration=None, title="Error — copy and send to the AI")
        query_info = self._query_panel(user_query, user_query, None, steps, errors, {}, False, None, lang=lang)
        return (
            query_info,
            self._stats_html(steps, lang=lang),
            pd.DataFrame(),
            pd.DataFrame(),
            "",
            "\n\n".join(errors) + "\n\n" + (tb or ""),
            pd.DataFrame(),
            {},
            user_query,
        )

    # -- panels ------------------------------------------------------------

    def _query_panel(self, user_query, search_query, rewrite, steps, errors, judge_map, judge_enabled, consensus_verdict, lang="ar"):
        return render_query_panel(user_query, search_query, rewrite, errors, judge_map, judge_enabled, consensus_verdict, lang=lang)

    def _stats_html(self, steps: list[Stat], lang="ar") -> str:
        return render_pipeline(steps, lang=lang)


@dataclass
class Settings:
    data_dir: Path
    device_embedder: str
    device_reranker: str
    device_judge: str = "cpu"
    warmup: bool = True


def positional_error(e: Exception) -> str:
    tb = traceback.format_exc()
    lines = [line.strip() for line in tb.splitlines() if line.strip()]
    return lines[-1] if lines else str(e)


# ---------------------------------------------------------------------------
# HTML Rendering Helpers
# ---------------------------------------------------------------------------


def esc(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def verdict_badge_class(verdict: str) -> str:
    return {
        "answers-it": "good",
        "does-not-answer": "bad",
        "related-only": "warn",
        "unrelated": "bad",
    }.get(verdict, "neutral")


def verdict_badge(verdict: str) -> str:
    label = VERDICT_LABELS.get(verdict, verdict or "unknown")
    icon = VERDICT_ICONS.get(verdict, "?")
    return (
        f'<span class="fq-badge {verdict_badge_class(verdict)}" title="{esc(verdict)}">'
        f"<b>{icon}</b>{esc(label)}</span>"
    )


def consensus_badge_class(consensus: str) -> str:
    return {
        CONSENSUS_AGREE: "good",
        CONSENSUS_PARTIAL: "warn",
        CONSENSUS_DISAGREE: "bad",
    }.get(consensus, "neutral")


def _safe_url(url: Any) -> str:
    url = str(url or "").strip()
    return url if re.match(r"^https?://", url, re.IGNORECASE) else ""


def _fmt_conf(conf: Any) -> str:
    if isinstance(conf, (int, float)):
        return f"{conf:.2f}"
    return esc(conf) if conf else "—"


def render_markdown_safe(text: str) -> str:
    """Render LLM markdown (headings, bold, lists) to HTML with raw HTML disabled."""
    from markdown_it import MarkdownIt

    return MarkdownIt("commonmark", {"html": False}).render(text or "")


def render_consensus_box(consensus: ConsensusVerdict | None, summary: str | None, lang: str = "ar") -> str:
    if not consensus and not summary:
        return ""

    parts = []
    if consensus and consensus.consensus != "unknown":
        badge_cls = consensus_badge_class(consensus.consensus)
        if lang == "ar":
            badge_lbl = CONSENSUS_LABELS_AR.get(consensus.consensus, consensus.consensus)
        else:
            badge_lbl = CONSENSUS_LABELS.get(consensus.consensus, consensus.consensus)
        icon = CONSENSUS_ICONS.get(consensus.consensus, "")
        conf_str = f" · confidence {_fmt_conf(consensus.confidence)}" if (lang == "en" and consensus.confidence is not None) else (
            f" · مستوى الثقة {_fmt_conf(consensus.confidence)}" if consensus.confidence is not None else ""
        )
        parts.append(
            f'<div class="fq-consensus-head">'
            f'<span class="fq-badge {badge_cls}"><b>{icon}</b> {esc(badge_lbl)}</span>'
            f'<span class="fq-conf">{conf_str}</span>'
            f'</div>'
        )
        if consensus.explanation:
            parts.append(
                f'<div class="fq-consensus-exp fq-ar" dir="rtl">{esc(consensus.explanation)}</div>'
            )

    if summary and summary.strip():
        parts.append(
            f'<div class="fq-summary-block">'
            f'<div class="fq-label">{I18N[lang]["summary_title"]}</div>'
            f'<div class="fq-summary-text fq-ar fq-md" dir="rtl">{render_markdown_safe(summary)}</div>'
            f'</div>'
        )

    if not parts:
        return ""
    return '<section class="fq-consensus-box">' + "".join(parts) + "</section>"


def render_query_panel(
    user_query, search_query, rewrite, errors, judge_map, judge_enabled, consensus_verdict=None, lang="ar"
) -> str:
    asked_lbl = "سألت" if lang == "ar" else "You asked"
    searched_lbl = "تم البحث بصيغة" if lang == "ar" else "Searched as"
    rows = [
        f'<div class="fq-qrow"><span class="fq-qlabel">{asked_lbl}</span>'
        f'<span class="fq-qtext" dir="auto">{esc(user_query)}</span></div>'
    ]
    if (search_query or "").strip() != (user_query or "").strip():
        rows.append(
            f'<div class="fq-qrow"><span class="fq-qlabel">{searched_lbl}</span>'
            f'<span class="fq-qtext fq-ar" dir="auto">{esc(search_query)}</span></div>'
        )

    chips = []
    if rewrite is not None:
        chips.append(f'<span class="fq-chip">{esc(rewrite.language)}</span>')
        rewrite_lbl = ("تمت إعادة الصياغة" if rewrite.changed else "كما هو") if lang == "ar" else ("rewritten" if rewrite.changed else "kept as is")
        chips.append(f'<span class="fq-chip">{rewrite_lbl}</span>')

    if consensus_verdict and consensus_verdict.consensus != "unknown":
        cls_name = consensus_badge_class(consensus_verdict.consensus)
        icon = CONSENSUS_ICONS.get(consensus_verdict.consensus, "")
        c_lbl = CONSENSUS_LABELS_AR.get(consensus_verdict.consensus, consensus_verdict.consensus) if lang == "ar" else CONSENSUS_LABELS.get(consensus_verdict.consensus, consensus_verdict.consensus)
        chips.append(f'<span class="fq-pill {cls_name}">{icon} {c_lbl}</span>')

    if judge_enabled and judge_map:
        counts: dict[str, int] = {}
        for j in judge_map.values():
            counts[j.verdict] = counts.get(j.verdict, 0) + 1
        for v in VERDICTS:
            if counts.get(v):
                v_lbl = VERDICT_LABELS_AR[v] if lang == "ar" else VERDICT_LABELS[v]
                chips.append(
                    f'<span class="fq-pill {verdict_badge_class(v)}">'
                    f"{VERDICT_ICONS[v]} {counts[v]} · {v_lbl}</span>"
                )
    if chips:
        rows.append('<div class="fq-qchips">' + "".join(chips) + "</div>")

    if errors:
        err_title = f"{len(errors)} مشكلة" if lang == "ar" else f"{len(errors)} issue{'s' if len(errors) != 1 else ''}"
        err_hint = "تفاصيل الخطأ الكاملة في تبويب <b>التشخيص</b>." if lang == "ar" else "Full traceback is in the <b>Debug</b> tab."
        items = "".join(f"<li>{esc(e)}</li>" for e in errors)
        rows.append(
            '<div class="fq-alert" role="alert">'
            f'<div class="fq-alert-title">{err_title}</div>'
            f"<ul>{items}</ul>"
            f'<div class="fq-alert-hint">{err_hint}</div></div>'
        )
    return '<section class="fq-query-panel">' + "".join(rows) + "</section>"


def render_pipeline(steps: list[Stat], lang: str = "ar") -> str:
    real = [s for s in steps if s.label != "Total"]
    total = next((s for s in steps if s.label == "Total"), None)
    total_s = total.seconds if total else sum(s.seconds for s in real)
    denom = total_s if total_s > 0 else 1.0
    tok_in = sum(s.prompt for s in real)
    tok_out = sum(s.completion for s in real)
    cost = sum(s.cost for s in real)
    aborted = bool(total and total.note == "aborted")

    lbl_time = "إجمالي الوقت" if lang == "ar" else "Total time"
    lbl_e2e = ("ملغى" if aborted else "كامل المعالجة") if lang == "ar" else ("aborted" if aborted else "end to end")
    lbl_tokens = "التوكنز" if lang == "ar" else "Tokens"
    lbl_tok_sub = f"{tok_in:,} مدخلات · {tok_out:,} مخرجات" if lang == "ar" else f"{tok_in:,} in · {tok_out:,} out"
    lbl_cost = "التكلفة التقديرية" if lang == "ar" else "Est. cost"
    lbl_cost_sub = "نداءات LLM فقط" if lang == "ar" else "LLM calls only"

    kpis = (
        f'<div class="fq-kpi"><div class="fq-kpi-label">{lbl_time}</div>'
        f'<div class="fq-kpi-value">{total_s:.2f}<small>s</small></div>'
        f'<div class="fq-kpi-sub">{lbl_e2e}</div></div>'
        f'<div class="fq-kpi"><div class="fq-kpi-label">{lbl_tokens}</div>'
        f'<div class="fq-kpi-value">{tok_in + tok_out:,}</div>'
        f'<div class="fq-kpi-sub">{lbl_tok_sub}</div></div>'
        f'<div class="fq-kpi"><div class="fq-kpi-label">{lbl_cost}</div>'
        f'<div class="fq-kpi-value">{f"${cost:.6f}" if cost > 0 else "—"}</div>'
        f'<div class="fq-kpi-sub">{lbl_cost_sub}</div></div>'
    )

    items = []
    cursor = 0.0
    for i, s in enumerate(real, 1):
        state = "fail" if s.note == "failed" else "off" if s.note in ("disabled", "no candidates", "insufficient results") else "ok"
        left = cursor / denom * 100
        width = max(s.seconds / denom * 100, 0.8) if s.seconds > 0 else 0
        cursor += s.seconds
        meta = []
        if s.model:
            meta.append(f'<span class="fq-chip mono">{esc(s.model)}</span>')
        if s.note:
            meta.append(f'<span class="fq-chip note {state}">{esc(s.note)}</span>')
        meta_html = "".join(meta)

        items.append(
            f'<li class="fq-pstep {state}">'
            f'<div class="fq-step-num">{i}</div>'
            f'<div class="fq-step-main">'
            f'<div class="fq-step-head"><div class="fq-step-name">{esc(s.label)}{meta_html}</div>'
            f'<div class="fq-step-time">{s.seconds:.2f}s</div></div>'
            f'<div class="fq-track"><span class="fq-fill" style="left:{left:.2f}%;width:{width:.2f}%"></span></div>'
            f"</div></li>"
        )

    return (
        f'<section class="fq-pipe">'
        f'<div class="fq-kpis">{kpis}</div>'
        f'<ol class="fq-psteps">{"".join(items)}</ol>'
        f"</section>"
    )


def error_modal_html(errors: list[str], user_query: str, steps: list[Stat], tb: str | None = None) -> str:
    body = [f"User query:\n  {user_query}\n\nErrors encountered:\n"]
    for e in errors:
        body.append(f"- {e}\n")
    if tb:
        body.append(f"\nTraceback:\n{tb}\n")
    return "\n".join(body)


def fatwa_card(
    row: pd.Series,
    judge: FatwaJudgement | None = None,
    rank: int = 0,
    is_true_label: bool = False,
    lang: str = "ar",
) -> str:
    categories = row.get("categories_list", []) or []
    tags = "".join(f'<span class="fq-cat fq-ar">{esc(cat)}</span>' for cat in categories)

    link = _safe_url(row.get("links", ""))
    orig_text = I18N[lang]["original_link"]
    link_html = (
        f'<a class="fq-btn primary" href="{esc(link)}" target="_blank" rel="noopener noreferrer">{esc(orig_text)}</a>'
        if link
        else ""
    )

    r_rank = int(row.get("rerank_rank", rank or 1))
    e_rank = int(row.get("retrieval_rank", 1))
    delta = e_rank - r_rank
    move = ""
    if delta:
        move = f'<span class="fq-m move {"up" if delta > 0 else "down"}">{"▲" if delta > 0 else "▼"} {abs(delta)}</span>'

    edge = f" v-{verdict_badge_class(judge.verdict)}" if judge is not None else ""
    top = " top" if rank == 1 else ""
    best = f'<span class="fq-best">{I18N[lang]["best_match"]}</span>' if rank == 1 else ""
    open_attr = " open" if rank == 1 else ""  # only the best match starts expanded
    tl_class = " is-true-label" if is_true_label else ""

    scholar_label = esc(row.get("source", "Scholar"))
    fid = int(row['fatwa_id'])
    title_esc = esc(row['title']).replace("'", "\\'").replace('"', "&quot;")

    # Verdict badge (on the left of the sheikh's name in same round button style)
    verdict_badge_html = ""
    if judge is not None and judge.verdict:
        v_cls = verdict_badge_class(judge.verdict)
        v_icon = VERDICT_ICONS.get(judge.verdict, "•")
        v_text = I18N[lang]["verdict_answers"] if judge.verdict == "answers-it" else (
            I18N[lang]["verdict_does_not_answer"] if judge.verdict == "does-not-answer" else I18N[lang]["verdict_unknown"]
        )
        verdict_badge_html = f'<span class="fq-badge {v_cls}"><b>{v_icon}</b> {v_text}</span>'

    scholar_badge_html = f'<span class="fq-scholar-badge">{scholar_label}</span>'

    q_label = I18N[lang]["question_label"]
    ans_label = I18N[lang]["full_answer"]
    copy_label = I18N[lang]["copy_answer"]

    return f"""
    <article class="fq-card{top}{edge}{tl_class}">
      <header class="fq-card-head">
        <div class="fq-rank">{r_rank}</div>
        <div class="fq-card-main">
          <h3 class="fq-card-title fq-ar" dir="rtl">{esc(row['title'])}</h3>
          <div class="fq-card-meta">
            {best}
            {verdict_badge_html}
            {scholar_badge_html}
            {tags}
          </div>
        </div>
        <div class="fq-card-votes">
          <button type="button" class="fq-vote-btn up" onclick="openFeedbackModal('{fid}', 'up', '{title_esc}')" title="Answers it" aria-label="Answers it"></button>
          <button type="button" class="fq-vote-btn down" onclick="openFeedbackModal('{fid}', 'down', '{title_esc}')" title="Does not answer" aria-label="Does not answer"></button>
        </div>
      </header>
      <div class="fq-qblock"><div class="fq-label">{q_label}</div>
        <div class="fq-ar fq-question" dir="rtl">{esc(row['question'])}</div></div>
      <details class="fq-det"{open_attr}>
        <summary><span>{ans_label}</span></summary>
        <div class="fq-ar fq-answer-text" dir="rtl">{esc(row['answer'])}</div>
      </details>
      <div class="fq-card-foot">
        <div class="fq-metrics">
          <span class="fq-m" title="Fatwa id">id {fid}</span>
          <span class="fq-m" title="Embedding cosine score and retrieval rank">embed {float(row['embedding_score']):.3f} <i>#{e_rank}</i></span>
          <span class="fq-m" title="Reranker score">rerank {float(row.get('reranker_score', 0.0)):.3f}</span>{move}
        </div>
        <div class="fq-actions">
          <button type="button" class="fq-btn" onclick="{COPY_JS}">{copy_label}</button>{link_html}
        </div>
      </div>
    </article>
    """


def render_results_cards(
    display_rows: pd.DataFrame,
    judge_map: dict[int, FatwaJudgement] | None = None,
    judge_enabled: bool = False,
    consensus_box_html: str = "",
    lang: str = "ar",
) -> str:
    if display_rows is None or not len(display_rows):
        return (consensus_box_html + "\n" if consensus_box_html else "") + f'<p class="empty">{I18N[lang]["empty_failed"]}</p>'

    prefix = (consensus_box_html + "\n" if consensus_box_html else "")

    if not judge_enabled or not judge_map:
        cards_html = "\n".join(
            fatwa_card(row, None, int(row.get("rerank_rank", idx + 1)), lang=lang)
            for idx, (_, row) in enumerate(display_rows.iterrows())
        )
        return prefix + cards_html

    # Partition into True Labels (answers-it) and Other Candidates
    true_labels = []
    other_candidates = []

    for idx, (_, row) in enumerate(display_rows.iterrows()):
        fid = int(row.fatwa_id)
        j = judge_map.get(fid)
        rank = int(row.get("rerank_rank", idx + 1))
        if j and j.verdict == "answers-it":
            true_labels.append((row, j, rank))
        else:
            other_candidates.append((row, j, rank))

    sections = []

    if true_labels:
        # Group 1: Verified Direct Answers
        hdr_1 = (
            f'<div class="fq-section-header">'
            f'<div class="fq-sect-title">{I18N[lang]["sect_direct_answers"]}</div>'
            f'<span class="fq-badge good"><b>✓</b> {len(true_labels)}</span>'
            f'</div>'
        )
        cards_1 = "\n".join(
            fatwa_card(row, j, rank, is_true_label=True, lang=lang)
            for row, j, rank in true_labels
        )
        sections.append(hdr_1 + "\n" + cards_1)

        if other_candidates:
            # Group 2: Other Candidates
            hdr_2 = (
                f'<div class="fq-section-header fq-sect-other">'
                f'<div class="fq-sect-title">{I18N[lang]["sect_other_candidates"]}</div>'
                f'<span class="fq-badge neutral">{len(other_candidates)}</span>'
                f'</div>'
            )
            cards_2 = "\n".join(
                fatwa_card(row, j, rank, is_true_label=False, lang=lang)
                for row, j, rank in other_candidates
            )
            sections.append(hdr_2 + "\n" + cards_2)
    else:
        # 0 True Labels: show section header with count and all candidates
        hdr_0 = (
            f'<div class="fq-section-header">'
            f'<div class="fq-sect-title">{I18N[lang]["sect_all_candidates"]}</div>'
            f'<span class="fq-badge neutral">{len(display_rows)}</span>'
            f'</div>'
        )
        cards_0 = "\n".join(
            fatwa_card(row, judge_map.get(int(row.fatwa_id)), int(row.get("rerank_rank", idx + 1)), is_true_label=False, lang=lang)
            for idx, (_, row) in enumerate(display_rows.iterrows())
        )
        sections.append(hdr_0 + "\n" + cards_0)

    return prefix + "\n".join(sections)


def render_empty_results(lang: str = "ar") -> str:
    title = I18N[lang]["empty_results_title"]
    desc = I18N[lang]["empty_results_desc"]
    return (
        f'<div class="fq-empty"><div class="fq-empty-title">{title}</div>'
        f'<div class="fq-empty-text">{desc}</div></div>'
    )


def render_loading_results(lang: str = "ar") -> str:
    title = I18N[lang]["loading_results_title"]
    return (
        f'<div class="fq-empty"><div class="fq-spin" style="margin:0 auto 12px"></div>'
        f'<div class="fq-empty-title">{title}</div></div>'
    )


def render_brand_html(lang: str, n_fatwas: int, n_scholars: int) -> str:
    title = I18N[lang]["title"]
    tagline = I18N[lang]["tagline_fmt"].format(n=n_fatwas, scholars=n_scholars)
    return (
        f"<div class='fq-brand'>{BRAND_MARK}<div><div class='fq-name'>{title}</div>"
        f"<div class='fq-tagline'>{tagline}</div></div></div>"
    )


def render_sidebar_header(lang: str) -> str:
    title = I18N[lang]["settings_title"]
    sub = I18N[lang]["settings_sub"]
    return (
        f"<div class='fq-side-title'>{title}</div>"
        f"<div class='fq-side-sub'>{sub}</div>"
    )


def render_modal_html() -> str:
    return """
    <div id="fq-feedback-modal" class="fq-modal-overlay" style="display:none;" onclick="if(event.target===this)closeFeedbackModal()">
      <div class="fq-modal-dialog">
        <div class="fq-modal-head">
          <h4 id="fq-modal-title"></h4>
          <button type="button" class="fq-modal-close" onclick="closeFeedbackModal()">&times;</button>
        </div>
        <div class="fq-modal-body">
          <p id="fq-modal-desc"></p>
          <div id="fq-modal-fatwa-info" class="fq-modal-fatwa-info"></div>
          <textarea id="fq-modal-comment" class="fq-modal-input" rows="3"></textarea>
        </div>
        <div class="fq-modal-foot">
          <button type="button" class="fq-btn-cancel" onclick="closeFeedbackModal()" id="fq-modal-cancel-btn">إلغاء</button>
          <button type="button" class="fq-btn primary" onclick="submitFeedbackModal()" id="fq-modal-submit-btn">إرسال التقييم</button>
        </div>
      </div>
    </div>
    <div id="fq-toast" class="fq-toast"></div>
    """


def dataframe_for_display(rows: pd.DataFrame, judge_map: dict[int, FatwaJudgement] | None = None) -> pd.DataFrame:
    if not len(rows):
        return pd.DataFrame()
    out = pd.DataFrame()
    if "rerank_rank" in rows.columns:
        out["Rerank #"] = rows["rerank_rank"]
    if "retrieval_rank" in rows.columns:
        out["Retrieval #"] = rows["retrieval_rank"]
    if "source" in rows.columns:
        out["Scholar"] = rows["source"]
    out["Title"] = rows["title"]
    if "reranker_score" in rows.columns:
        out["Rerank score"] = rows["reranker_score"].map(lambda v: f"{float(v):.3f}")
    if "embedding_score" in rows.columns:
        out["Cosine"] = rows["embedding_score"].map(lambda v: f"{float(v):.3f}")
    if judge_map is not None:
        out["Verdict"] = rows["fatwa_id"].map(
            lambda fid: (judge_map[int(fid)].verdict if int(fid) in judge_map else "—")
        )
    return out


def df_to_html(df: Any) -> str:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return EMPTY_TABLE
    headers = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = row[c]
            cls = " fq-ar" if c in ("Title", "title") else ""
            cells.append(f"<td class='{cls}'>{esc(val)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<div class='fq-tbl-wrap'><table class='fq-tbl'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Theme + CSS
# ---------------------------------------------------------------------------

_GREEN = gr.themes.Color(
    name="fq_green",
    c50="#eef7f3", c100="#d8ece4", c200="#b2d9c9", c300="#82c0aa", c400="#51a28a",
    c500="#2e8670", c600="#1d6c5a", c700="#17594a", c800="#13483c", c900="#0f3a31", c950="#08231d",
)

THEME = gr.themes.Base(
    primary_hue=_GREEN,
    secondary_hue=_GREEN,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.Font("Manrope"), gr.themes.Font("IBM Plex Sans Arabic"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.Font("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="var(--fq-bg)",
    body_background_fill_dark="var(--fq-bg)",
    button_primary_background_fill="var(--fq-accent)",
    button_primary_background_fill_hover="var(--fq-accent-ink)",
    button_primary_background_fill_dark="var(--fq-accent)",
    button_primary_background_fill_hover_dark="var(--fq-accent-ink)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#06130f",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --fq-bg:#f4f1e8; --fq-surface:#fffdf8; --fq-surface-2:#f1ede1; --fq-ink:#1b2724; --fq-muted:#6b7570;
  --fq-line:#e3dccb; --fq-accent:#17715c; --fq-accent-ink:#0f5444; --fq-accent-soft:#e1efe9;
  --fq-gold:#b3832a; --fq-gold-soft:#f6ebcc;
  --fq-good:#17794f; --fq-good-soft:#dcf1e6; --fq-warn:#94650a; --fq-warn-soft:#faedc9;
  --fq-bad:#b03a3a; --fq-bad-soft:#f7dfdb; --fq-neutral:#5f6a66; --fq-neutral-soft:#e9e6dc;
  --fq-shadow:0 1px 2px rgba(50,38,10,.05), 0 10px 28px -14px rgba(50,38,10,.16);
  --fq-ar:'Amiri','Noto Naskh Arabic','Traditional Arabic','Geeza Pro',serif;
  --fq-sans:'Manrope','IBM Plex Sans Arabic',system-ui,sans-serif;
  --fq-mono:'JetBrains Mono',ui-monospace,Consolas,monospace;
}
.dark {
  --fq-bg:#0d1412; --fq-surface:#151f1c; --fq-surface-2:#1b2824; --fq-ink:#e7eeeb; --fq-muted:#8fa09a;
  --fq-line:#26352f; --fq-accent:#3dc3a0; --fq-accent-ink:#6fdbbf; --fq-accent-soft:#143229;
  --fq-gold:#e2b658; --fq-gold-soft:#34290f;
  --fq-good:#58d69a; --fq-good-soft:#12301f; --fq-warn:#e8bb55; --fq-warn-soft:#352a0c;
  --fq-bad:#f08a80; --fq-bad-soft:#3a1816; --fq-neutral:#a2b0aa; --fq-neutral-soft:#222e2a;
  --fq-shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 28px -14px rgba(0,0,0,.6);
}

body, .gradio-container { background:var(--fq-bg) !important; font-family:var(--fq-sans) !important; color:var(--fq-ink); }
.gradio-container { max-width:1080px !important; margin:0 auto !important; padding-top:10px !important; }
footer { display:none !important; }
.gradio-container .block, .gradio-container .form { box-shadow:none; }
.fq-ar { font-family:var(--fq-ar) !important; }

/* ---- top bar ---- */
#fq-top { align-items:center !important; gap:12px; margin:6px 0 4px; flex-wrap:nowrap !important; }
#fq-top > div { min-width:0 !important; }
.fq-brand { display:flex; align-items:center; gap:14px; }
.fq-mark { width:44px; height:44px; color:var(--fq-accent); flex:none; }
.fq-mark .gold { fill:var(--fq-gold); stroke:none; }
.fq-name { font-weight:800; font-size:1.45rem; letter-spacing:-.02em; line-height:1.1; color:var(--fq-ink); }
.fq-tagline { font-size:.82rem; color:var(--fq-muted); margin-top:3px; }
#fq-settings-btn, #fq-theme-btn, #fq-lang-btn { flex:none !important; width:auto !important; min-width:0 !important; }
#fq-settings-btn, #fq-theme-btn, #fq-lang-btn { border-radius:999px; background:var(--fq-surface); border:1px solid var(--fq-line);
  color:var(--fq-ink) !important; font-weight:600; padding:9px 16px !important; font-size:.9rem !important; box-shadow:var(--fq-shadow); cursor:pointer; }
#fq-settings-btn:hover, #fq-theme-btn:hover, #fq-lang-btn:hover { border-color:var(--fq-accent); color:var(--fq-accent); }
.fq-side-title { font-weight:800; font-size:1.35rem; color:var(--fq-ink); margin-bottom:4px; }
.fq-side-sub { font-size:.85rem; color:var(--fq-muted); margin-bottom:14px; }

/* ---- search controls & mode selector ---- */
#fq-search-wrap { background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px;
  padding:6px; box-shadow:var(--fq-shadow); margin-top:10px; }
#fq-query, #fq-query *:not(button):not(svg):not(path) { background:transparent !important; border:none !important; box-shadow:none !important; }
#fq-search-wrap, #fq-search-wrap .styler { --block-background-fill:transparent; }
#fq-search-wrap .styler { background:transparent !important; }
#fq-search-wrap > div, #fq-search-wrap .block, #fq-search-wrap label { border:none !important; box-shadow:none !important; background:transparent !important; }
#fq-query textarea { font-family:var(--fq-ar) !important; font-size:1.35rem !important; line-height:1.7 !important;
  padding:12px 16px !important; background:transparent !important; border:none !important; box-shadow:none !important; }
#fq-query textarea::placeholder { font-family:var(--fq-sans); font-size:1rem; color:var(--fq-muted); }
#fq-query button { background:var(--fq-accent) !important; color:#fff !important; border-radius:14px !important; }
/* Search options strip: sits directly under the search box, no container of its own */
#fq-controls { margin:14px 6px 0 !important; gap:28px !important; align-items:flex-start !important;
  flex-wrap:wrap !important; background:transparent !important; border:none !important; }
#fq-controls > div, #fq-controls .block, #fq-controls .form { background:transparent !important; border:none !important;
  box-shadow:none !important; padding:0 !important; }
#fq-controls [data-testid="block-info"], #fq-controls .block-info, #fq-controls label > span:first-child:not(.ml-2) {
  font-size:.8rem !important; font-weight:600 !important; color:var(--fq-muted) !important; margin-bottom:6px !important; }
#fq-mode .wrap, #fq-scholars .wrap { gap:6px !important; flex-wrap:wrap !important; }
#fq-mode .wrap label, #fq-scholars .wrap label { position:relative; margin:0 !important; padding:6px 14px !important;
  border-radius:999px !important; border:1px solid var(--fq-line) !important; background:var(--fq-surface) !important;
  color:var(--fq-muted) !important; font-size:.86rem !important; font-weight:600 !important; box-shadow:none !important;
  cursor:pointer; transition:background .15s, color .15s, border-color .15s; }
#fq-mode .wrap label input, #fq-scholars .wrap label input { position:absolute; opacity:0; width:1px; height:1px; pointer-events:none; }
#fq-mode .wrap label span, #fq-scholars .wrap label span { margin:0 !important; padding:0 !important; }
#fq-mode .wrap label:hover, #fq-scholars .wrap label:hover { border-color:var(--fq-accent) !important; color:var(--fq-ink) !important; }
#fq-mode .wrap label.selected, #fq-mode .wrap label:has(input:checked),
#fq-scholars .wrap label.selected, #fq-scholars .wrap label:has(input:checked) {
  background:var(--fq-accent-soft) !important; border-color:var(--fq-accent) !important; color:var(--fq-accent-ink) !important; }
#fq-mode .wrap label:has(input:focus-visible), #fq-scholars .wrap label:has(input:focus-visible) {
  outline:2px solid var(--fq-accent); outline-offset:2px; }
#fq-controls .fq-count { min-width:180px !important; max-width:220px; }
#fq-controls .fq-count input[type=number] { background:var(--fq-surface) !important; border:1px solid var(--fq-line) !important; border-radius:8px !important; }

/* ---- generic ---- */
.fq-flat, .fq-flat.block { border:none !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }
.fq-hidden { display: none !important; }
.fq-chip { display:inline-block; padding:2px 10px; border-radius:999px; background:var(--fq-surface-2); color:var(--fq-muted);
  font-size:.76rem; font-weight:600; border:1px solid var(--fq-line); }
.fq-chip.mono { font-family:var(--fq-mono); font-weight:400; font-size:.72rem; }
.fq-chip.note.fail { background:var(--fq-bad-soft); color:var(--fq-bad); border-color:transparent; }
.fq-chip.note.off { opacity:.8; }
.fq-pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:700; }
.fq-pill.good,.fq-badge.good { background:var(--fq-good-soft); color:var(--fq-good); border:1px solid color-mix(in srgb, var(--fq-good) 30%, transparent); }
.fq-pill.warn,.fq-badge.warn { background:var(--fq-warn-soft); color:var(--fq-warn); border:1px solid color-mix(in srgb, var(--fq-warn) 30%, transparent); }
.fq-pill.bad,.fq-badge.bad { background:var(--fq-bad-soft); color:var(--fq-bad); border:1px solid color-mix(in srgb, var(--fq-bad) 30%, transparent); }
.fq-pill.neutral,.fq-badge.neutral { background:var(--fq-neutral-soft); color:var(--fq-neutral); border:1px solid var(--fq-line); }
.fq-badge { display:inline-flex; align-items:center; gap:6px; padding:3px 12px; border-radius:999px; font-size:.85rem; font-weight:700; }
.fq-badge b { font-size:.95rem; }
.fq-label { font-size:.72rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); margin:0 0 6px; }

/* ---- consensus & summary box ---- */
.fq-consensus-box { margin:16px 0 20px; padding:20px 24px; border-radius:20px; background:var(--fq-surface);
  border:1.5px solid var(--fq-line); box-shadow:var(--fq-shadow); }
.fq-consensus-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
.fq-consensus-exp { font-size:1.15rem; line-height:1.7; color:var(--fq-ink); margin-bottom:14px; }
.fq-summary-block { margin-top:14px; padding-top:14px; border-top:1px dashed var(--fq-line); }
.fq-summary-text { font-size:1.15rem; line-height:1.8; color:var(--fq-ink); }
.fq-md h1, .fq-md h2, .fq-md h3, .fq-md h4 { font-size:1.15rem; font-weight:700; margin:14px 0 6px; }
.fq-md p { margin:0 0 8px; } .fq-md ul, .fq-md ol { margin:0 0 8px; padding-inline-start:22px; }
.fq-md hr { border:0; border-top:1px dashed var(--fq-line); margin:12px 0; }

.fq-scholar-badge { display:inline-flex; align-items:center; gap:5px; padding:3px 12px; border-radius:999px;
  background:var(--fq-accent-soft); color:var(--fq-accent-ink); font-size:.85rem; font-weight:700; border:1px solid color-mix(in srgb, var(--fq-accent) 30%, transparent); }

/* ---- query panel ---- */
.fq-query-panel { margin:16px 2px 6px; padding:14px 18px; border-radius:16px; background:var(--fq-surface-2); border:1px solid var(--fq-line); }
.fq-qrow { display:flex; gap:12px; align-items:baseline; padding:3px 0; }
.fq-qlabel { flex:none; width:92px; font-size:.7rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:var(--fq-muted); }
.fq-qtext { font-size:1rem; font-weight:500; }
.fq-qtext.fq-ar { font-size:1.3rem; }
.fq-qchips { display:flex; gap:8px; flex-wrap:wrap; margin:9px 0 0 104px; }
.fq-alert { margin-top:12px; padding:12px 16px; border-radius:12px; background:var(--fq-bad-soft); color:var(--fq-bad); }

/* ---- tabs ---- */
#fq-tabs { margin-top:14px; }
#fq-tabs button[role=tab] { font-weight:700; font-size:.95rem; }
#fq-tabs button[role=tab].selected { color:var(--fq-accent) !important; }

/* ---- result cards ---- */
.fq-card { position:relative; background:var(--fq-surface); border:1px solid var(--fq-line); border-radius:20px;
  padding:20px 24px 16px; margin:14px 0; box-shadow:var(--fq-shadow); --edge:var(--fq-accent); }
.fq-card::before { content:""; position:absolute; left:0; top:22px; bottom:22px; width:4px; border-radius:0 4px 4px 0; background:var(--edge); }
.fq-card.v-good { --edge:var(--fq-good); } .fq-card.v-warn { --edge:var(--fq-warn); } .fq-card.v-bad { --edge:var(--fq-bad); opacity:.92; }
.fq-card.top { border-color:color-mix(in srgb, var(--fq-gold) 55%, var(--fq-line)); }
.fq-card-head { display:flex; gap:16px; align-items:flex-start; }
.fq-rank { flex:none; width:38px; height:38px; border-radius:12px; background:var(--fq-accent-soft); color:var(--fq-accent-ink);
  display:grid; place-items:center; font-weight:800; font-size:1.05rem; }
.fq-card-main { flex:1; min-width:0; }
.fq-card-title { margin:0 0 8px !important; font-size:1.55rem !important; line-height:1.55 !important; font-weight:700; color:var(--fq-ink); text-align:right; }
.fq-card-meta { display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; }
.fq-best { font-size:.74rem; font-weight:800; color:var(--fq-gold); background:var(--fq-gold-soft); padding:2px 10px; border-radius:999px; }
.fq-cat { display:inline-block; padding:0 11px; border-radius:999px; background:var(--fq-surface-2); color:var(--fq-muted); font-size:0.92rem; line-height:1.7; border:1px solid var(--fq-line); }

/* Card votes (pure icon buttons in corner) */
.fq-card-votes { display:flex; gap:6px; align-items:center; margin-left:auto; flex-shrink:0; }
.fq-vote-btn { width:36px; height:36px; border-radius:50%; border:1px solid var(--fq-line);
  background:var(--fq-surface); color:var(--fq-ink); display:inline-flex; align-items:center; justify-content:center;
  cursor:pointer; color:var(--fq-muted); transition:all 0.2s cubic-bezier(0.4, 0, 0.2, 1); box-shadow:0 1px 2px rgba(0,0,0,0.04); }
.fq-vote-btn:hover { transform:scale(1.12); }
.fq-vote-btn.up:hover { background:var(--fq-good-soft); border-color:var(--fq-good); color:var(--fq-good); }
.fq-vote-btn.down:hover { background:var(--fq-bad-soft); border-color:var(--fq-bad); color:var(--fq-bad); }
.fq-vote-btn { display:inline-flex !important; padding:0 !important; }
.fq-vote-btn::before { content:"" !important; display:block !important; flex:none !important; width:17px !important; min-width:17px !important; height:17px !important; margin:0 !important; background:currentColor !important;
  -webkit-mask:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M7 10v11H3V10h4zm0 0 4-8a2.5 2.5 0 0 1 2.5 2.5V8h5.3a2 2 0 0 1 2 2.3l-1.3 8.5a2 2 0 0 1-2 1.7H7'/></svg>") center/contain no-repeat; mask:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'><path d='M7 10v11H3V10h4zm0 0 4-8a2.5 2.5 0 0 1 2.5 2.5V8h5.3a2 2 0 0 1 2 2.3l-1.3 8.5a2 2 0 0 1-2 1.7H7'/></svg>") center/contain no-repeat; }
.fq-vote-btn.down::before { transform:rotate(180deg); }
.fq-vote-btn:focus-visible { outline:2px solid var(--fq-accent); outline-offset:2px; }

.fq-qblock { margin:14px 0 10px; }
.fq-question { font-size:1.2rem; line-height:1.65; color:var(--fq-ink); text-align:right; }
.fq-det summary { cursor:pointer; font-weight:700; font-size:.86rem; color:var(--fq-muted); padding:6px 0; }
.fq-det summary:hover { color:var(--fq-accent); }
.fq-answer-text { padding:12px 14px; border-radius:12px; background:var(--fq-surface-2); font-size:1.2rem; line-height:1.8; color:var(--fq-ink); text-align:right; }

.fq-card-foot { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:14px; padding-top:12px; border-top:1px solid var(--fq-line); flex-wrap:wrap; }
.fq-metrics { display:flex; gap:10px; align-items:center; flex-wrap:wrap; font-size:.8rem; }
.fq-m { color:var(--fq-muted); font-variant-numeric:tabular-nums; }
.fq-m.move.up { color:var(--fq-good); font-weight:700; }
.fq-m.move.down { color:var(--fq-bad); font-weight:700; }
.fq-actions { display:flex; gap:8px; align-items:center; }
.fq-btn { border-radius:999px; padding:5px 14px; font-size:.82rem; font-weight:600; border:1px solid var(--fq-line); background:var(--fq-surface); color:var(--fq-ink); cursor:pointer; }
.fq-btn:hover { border-color:var(--fq-accent); color:var(--fq-accent); }
.fq-btn.primary { background:var(--fq-accent); color:#fff; border-color:var(--fq-accent); text-decoration:none; display:inline-block; }
.fq-btn.primary:hover { background:var(--fq-accent-ink); border-color:var(--fq-accent-ink); }

/* pipeline */
.fq-pipe { padding:8px 0; }
.fq-kpis { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; margin-bottom:20px; }
.fq-kpi { padding:14px 18px; border-radius:16px; background:var(--fq-surface); border:1px solid var(--fq-line); }
.fq-kpi-label { font-size:.72rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; color:var(--fq-muted); }
.fq-kpi-value { font-size:1.8rem; font-weight:800; color:var(--fq-ink); margin:2px 0; }
.fq-kpi-sub { font-size:.8rem; color:var(--fq-muted); }
.fq-psteps { list-style:none; padding:0; margin:0; }
.fq-pstep { display:flex; gap:14px; align-items:center; padding:10px 0; border-bottom:1px solid var(--fq-line); }
.fq-step-num { width:28px; height:28px; border-radius:50%; background:var(--fq-surface-2); display:grid; place-items:center; font-weight:700; font-size:.85rem; }
.fq-step-main { flex:1; min-width:0; }
.fq-step-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:4px; }
.fq-step-name { font-weight:700; font-size:.92rem; display:flex; gap:8px; align-items:center; }
.fq-step-time { font-family:var(--fq-mono); font-size:.85rem; font-weight:600; color:var(--fq-muted); }
.fq-track { height:5px; border-radius:999px; background:var(--fq-surface-2); position:relative; overflow:hidden; }
.fq-fill { position:absolute; height:100%; border-radius:999px; background:var(--fq-accent); }
.fq-pstep.fail .fq-fill { background:var(--fq-bad); }

/* debug table */
.fq-tbl-wrap { overflow-x:auto; border-radius:14px; border:1px solid var(--fq-line); background:var(--fq-surface); }
.fq-tbl { width:100%; border-collapse:collapse; font-size:.86rem; text-align:left; }
.fq-tbl th, .fq-tbl td { padding:10px 14px; border-bottom:1px solid var(--fq-line); }
.fq-tbl th { background:var(--fq-surface-2); font-weight:800; text-transform:uppercase; font-size:.72rem; letter-spacing:.06em; color:var(--fq-muted); }
.fq-tbl td.fq-ar { text-align:right; font-family:var(--fq-ar); font-size:1.15rem; }

/* ---- results section headers (Claude's clean aesthetic) ---- */
.fq-section-header { display:flex; align-items:center; gap:10px; margin:20px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--fq-line); }
.fq-section-header.fq-sect-other { margin-top:32px; }
.fq-sect-title { font-size:1.15rem; font-weight:800; color:var(--fq-ink); margin:0; }

/* ---- feedback modal overlay & dialog ---- */
.fq-modal-overlay {
  position: fixed; inset: 0; background: rgba(0, 0, 0, 0.48); backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center;
  z-index: 99999; animation: fq-fade-in 0.2s ease;
}
@keyframes fq-fade-in { from { opacity: 0; } to { opacity: 1; } }
.fq-modal-dialog {
  background: var(--fq-surface); border: 1.5px solid var(--fq-line); border-radius: 20px;
  width: 90%; max-width: 460px; padding: 22px 24px; box-shadow: var(--fq-shadow);
  animation: fq-pop-in 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes fq-pop-in { from { transform: scale(0.92); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.fq-modal-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.fq-modal-head h4 { margin: 0; font-size: 1.2rem; font-weight: 800; color: var(--fq-ink); }
.fq-modal-close {
  background: transparent; border: none; font-size: 1.6rem; line-height: 1;
  cursor: pointer; color: var(--fq-muted); padding: 0 4px;
}
.fq-modal-close:hover { color: var(--fq-bad); }
.fq-modal-body p { margin: 0 0 10px; font-size: 0.95rem; color: var(--fq-ink); line-height: 1.55; }
.fq-modal-fatwa-info {
  font-size: 0.84rem; color: var(--fq-muted); margin-bottom: 14px; padding: 8px 12px;
  border-radius: 10px; background: var(--fq-surface-2); font-weight: 600;
}
.fq-modal-input {
  width: 100%; min-height: 75px; padding: 10px 12px; border-radius: 12px;
  border: 1px solid var(--fq-line); background: var(--fq-bg); color: var(--fq-ink);
  font-family: var(--fq-sans); font-size: 0.92rem; resize: vertical; margin-bottom: 16px;
  box-sizing: border-box;
}
.fq-modal-input:focus { outline: none; border-color: var(--fq-accent); }
.fq-modal-foot { display: flex; justify-content: flex-end; gap: 10px; }
.fq-btn-cancel {
  border-radius: 999px; padding: 6px 16px; font-size: .85rem; font-weight: 600;
  border: 1px solid var(--fq-line); background: var(--fq-surface); color: var(--fq-muted); cursor: pointer;
}
.fq-btn-cancel:hover { color: var(--fq-ink); border-color: var(--fq-ink); }

/* Toast */
.fq-toast {
  position: fixed; bottom: 28px; right: 28px; z-index: 100000; padding: 12px 22px;
  border-radius: 14px; background: var(--fq-accent); color: #ffffff; font-weight: 700;
  font-size: 0.95rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
  transform: translateY(120px); opacity: 0; pointer-events: none;
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease;
}
.fq-toast.show { transform: translateY(0); opacity: 1; }
"""

BRAND_MARK = """
<svg class="fq-mark" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="22" cy="22" r="21" stroke="currentColor" stroke-width="2"/>
  <path d="M22 6 L25 18 L37 22 L25 26 L22 38 L19 26 L7 22 L19 18 Z" class="gold"/>
  <circle cx="22" cy="22" r="3" fill="currentColor"/>
</svg>
"""

COPY_JS = """
(function(btn){
  var card = btn.closest('.fq-card');
  var txt = card ? card.querySelector('.fq-answer-text') : null;
  if(!txt) return;
  navigator.clipboard.writeText(txt.innerText.trim()).then(function(){
    var old = btn.innerText;
    var lang = window._currentLang || 'ar';
    btn.innerText = (lang === 'ar') ? 'تم النسخ!' : 'Copied!';
    setTimeout(function(){ btn.innerText = old; }, 1800);
  });
})(this);
"""

EMPTY_QUERY = '<div class="fq-empty-sm">Ask a question above to start searching.</div>'
EMPTY_RESULTS = (
    '<div class="fq-empty"><div class="fq-empty-title">Ready to search</div>'
    '<div class="fq-empty-text">Enter a question in Arabic or any language. Multiple scholar datasets are indexed and ready to compare.</div></div>'
)
EMPTY_PIPELINE = '<div class="fq-empty-sm">Pipeline statistics appear here after your first search.</div>'
EMPTY_TABLE = '<div class="fq-empty-sm">No data yet.</div>'
EMPTY_FAILED = '<div class="fq-empty-sm">Search produced no results.</div>'

LOADING_QUERY = (
    '<div class="fq-loading"><div class="fq-spin"></div>'
    '<span>Searching and synthesizing across scholarly fatwas…</span></div>'
)
LOADING_RESULTS = '<div class="fq-empty"><div class="fq-spin" style="margin:0 auto 12px"></div><div class="fq-empty-title">Retrieving & analyzing…</div></div>'

# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------


def build_ui(app: FatwaApp) -> gr.Blocks:
    n_fatwas = len(app.df) if hasattr(app, "df") else 0
    emb_dev = getattr(getattr(app, "s", None), "device_embedder", "?")
    rer_dev = getattr(getattr(app, "s", None), "device_reranker", "?")
    judge_dev = getattr(getattr(app, "s", None), "device_judge", "?")
    available_scholars = getattr(app, "available_scholars", ["Ibn Baz"])

    def key_chip(env: str, name: str) -> str:
        ok = bool(os.getenv(env))
        return f'<span class="fq-key {"ok" if ok else "no"}">{"✓" if ok else "✕"} {name}</span>'

    with gr.Blocks(title="Fatwa Search — Multi-Scholar RAG") as demo:
        gr.HTML(f"<style>{CSS}</style>", visible=False)
        sidebar_open = gr.State(False)
        lang_state = gr.State("ar")

        # ---- theme initialization & client actions script ------------------
        gr.HTML("""
        <script>
        (function() {
            window._activeFeedbackFid = null;
            window._activeFeedbackType = null;
            window._currentLang = 'ar';

            window._i18n = {
              ar: {
                modal_up_title: "تأكيد صحة الإجابة",
                modal_up_desc: "هل هذه الفتوى تجيب مباشرة وبدقة على سؤالك؟ سيتم حفظها كإجابة معتمدة لتدريب نموذج لَيا.",
                modal_down_title: "الإبلاغ عن عدم دقة الإجابة",
                modal_down_desc: "هل هذه الفتوى لا تجيب على سؤالك أو صُنفت بالخطأ؟ سيتم حفظ تقريرك لتحسين وتدريب نموذج لَيا.",
                modal_comment_placeholder: "ملاحظة أو توضيح إضافي (اختياري)...",
                modal_submit: "إرسال التقييم",
                modal_cancel: "إلغاء",
                toast_success: "جزاكم الله خيراً، تم تسجيل تقييمك لتدريب لَيا.",
              },
              en: {
                modal_up_title: "Confirm Direct Answer",
                modal_up_desc: "Does this fatwa directly and accurately answer your question? It will be saved as a verified answer to train Laya.",
                modal_down_title: "Report Inaccuracy",
                modal_down_desc: "Does this fatwa fail to answer your question or was it misclassified? Your report will help improve Laya.",
                modal_comment_placeholder: "Optional comment or explanation...",
                modal_submit: "Submit Feedback",
                modal_cancel: "Cancel",
                toast_success: "Thanks. Your feedback was saved for retraining Laya.",
              }
            };

            window.openFeedbackModal = function(fid, type, title) {
              window._activeFeedbackFid = fid;
              window._activeFeedbackType = type;
              
              var modal = document.getElementById('fq-feedback-modal');
              var mTitle = document.getElementById('fq-modal-title');
              var mDesc = document.getElementById('fq-modal-desc');
              var mInfo = document.getElementById('fq-modal-fatwa-info');
              var mComment = document.getElementById('fq-modal-comment');
              var mSubmit = document.getElementById('fq-modal-submit-btn');
              var mCancel = document.getElementById('fq-modal-cancel-btn');
              
              var lang = window._currentLang || 'ar';
              var texts = window._i18n[lang] || window._i18n['ar'];
              
              if (type === 'up') {
                mTitle.innerText = texts.modal_up_title;
                mDesc.innerText = texts.modal_up_desc;
              } else {
                mTitle.innerText = texts.modal_down_title;
                mDesc.innerText = texts.modal_down_desc;
              }
              
              mInfo.innerText = '#' + fid + ' - ' + (title || '');
              mComment.value = '';
              mComment.placeholder = texts.modal_comment_placeholder;
              mSubmit.innerText = texts.modal_submit;
              mCancel.innerText = texts.modal_cancel;
              
              if (modal) modal.style.display = 'flex';
            };

            window.closeFeedbackModal = function() {
              var modal = document.getElementById('fq-feedback-modal');
              if (modal) modal.style.display = 'none';
              window._activeFeedbackFid = null;
              window._activeFeedbackType = null;
            };

            window.submitFeedbackModal = function() {
              var fid = window._activeFeedbackFid;
              var type = window._activeFeedbackType;
              var comment = (document.getElementById('fq-modal-comment') ? document.getElementById('fq-modal-comment').value : '');
              
              var payload = JSON.stringify({ fid: fid, type: type, comment: comment });
              
              var box = document.querySelector('#fq-fb-payload textarea, #fq-fb-payload input');
              if (box) {
                box.value = payload;
                box.dispatchEvent(new Event('input', { bubbles: true }));
                box.dispatchEvent(new Event('change', { bubbles: true }));
              }
              var trigger = document.getElementById('fq-fb-trigger');
              if (trigger) {
                var btn = trigger.tagName === 'BUTTON' ? trigger : (trigger.querySelector('button') || trigger);
                if (btn) btn.click();
              }
              
              closeFeedbackModal();
              var lang = window._currentLang || 'ar';
              var toastMsg = (window._i18n[lang] && window._i18n[lang].toast_success) || 'Saved!';
              window.showToast(toastMsg);
            };

            window.showToast = function(msg) {
              var toast = document.getElementById('fq-toast');
              if (!toast) return;
              toast.innerText = msg;
              toast.classList.add('show');
              setTimeout(function() {
                toast.classList.remove('show');
              }, 3500);
            };

            document.addEventListener('keydown', function(e) {
              if (e.key === 'Escape') {
                window.closeFeedbackModal();
              }
            });

            function updateTheme() {
                var saved = null;
                try { saved = localStorage.getItem('fq-theme-mode'); } catch(e) {}
                var isDark = true;
                if (saved === 'light') {
                    isDark = false;
                } else if (saved === 'dark') {
                    isDark = true;
                } else {
                    isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
                }
                document.body.classList.toggle('dark', isDark);
                document.documentElement.classList.toggle('dark', isDark);
                var gApp = document.querySelector('gradio-app');
                if (gApp) gApp.classList.toggle('dark', isDark);
                var btn = document.getElementById('fq-theme-btn');
                if (btn) {
                    var b = btn.tagName === 'BUTTON' ? btn : (btn.querySelector('button') || btn);
                    b.innerText = isDark ? "Light" : "Dark";
                }
            }
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', updateTheme);
            } else {
                updateTheme();
            }
            setTimeout(updateTheme, 150);
            setTimeout(updateTheme, 600);
        })();
        </script>
        """, visible=False)

        # ---- top bar -------------------------------------------------------
        with gr.Row(elem_id="fq-top"):
            brand_html = gr.HTML(
                render_brand_html("ar", n_fatwas, len(available_scholars)),
                elem_classes="fq-flat",
            )
            lang_btn = gr.Button("English", elem_id="fq-lang-btn", size="sm", scale=0)
            theme_btn = gr.Button("Light", elem_id="fq-theme-btn", size="sm", scale=0)
            settings_btn = gr.Button(I18N["ar"]["settings_btn"], elem_id="fq-settings-btn", size="sm", scale=0)

        # ---- search box ----------------------------------------------------
        with gr.Group(elem_id="fq-search-wrap"):
            query = gr.Textbox(
                placeholder=I18N["ar"]["placeholder"],
                show_label=False,
                container=False,
                lines=1,
                max_lines=5,
                rtl=True,
                autofocus=True,
                submit_btn=True,
                elem_id="fq-query",
            )

        # ---- search options: a single compact strip under the search box ----
        with gr.Row(elem_id="fq-controls", equal_height=False):
            search_mode = gr.Radio(
                choices=SEARCH_MODES,
                value=SEARCH_MODE_PER_SCHOLAR,
                label=I18N["ar"]["search_mode_label"],
                scale=0,
                min_width=260,
                elem_id="fq-mode",
            )
            selected_scholars = gr.CheckboxGroup(
                choices=available_scholars,
                value=available_scholars,
                label=I18N["ar"]["scholars_label"],
                scale=1,
                elem_id="fq-scholars",
            )
            x_per_scholar = gr.Slider(
                1, 5, value=DEFAULT_X_PER_SCHOLAR, step=1,
                label=I18N["ar"]["x_per_scholar_label"],
                visible=True, scale=0, min_width=190, elem_classes="fq-count",
            )
            show_n = gr.Slider(
                1, 20, value=DEFAULT_SHOW_N, step=1,
                label=I18N["ar"]["show_n_label"],
                visible=False, scale=0, min_width=190, elem_classes="fq-count",
            )

        query_info = gr.HTML(I18N["ar"]["empty_query"], elem_classes="fq-flat")

        # ---- results -------------------------------------------------------
        with gr.Tabs(elem_id="fq-tabs"):
            with gr.Tab("Results"):
                results_html = gr.HTML(render_empty_results("ar"), elem_classes="fq-flat")
                
                # Hidden feedback trigger and payload for modal dialog submission
                fb_payload = gr.Textbox(value="", elem_id="fq-fb-payload", elem_classes=["fq-hidden"])
                fb_trigger = gr.Button("Submit Feedback", elem_id="fq-fb-trigger", elem_classes=["fq-hidden"])
                gr.HTML(render_modal_html())

            with gr.Tab("Pipeline"):
                stats_html = gr.HTML(EMPTY_PIPELINE, elem_classes="fq-flat")
            with gr.Tab("Debug"):
                gr.HTML(
                    "<div class='fq-sect-title'>Reranked candidates</div>"
                    "<div class='fq-sect-sub'>The fatwas that were shown/judged.</div>",
                    elem_classes="fq-flat",
                )
                rerank_html = gr.HTML(EMPTY_TABLE, elem_classes="fq-flat")
                gr.HTML(
                    "<div class='fq-sect-title'>Embedding retrieval</div>"
                    "<div class='fq-sect-sub'>Raw top-K by cosine similarity, before reranking.</div>",
                    elem_classes="fq-flat",
                )
                retrieval_html = gr.HTML(EMPTY_TABLE, elem_classes="fq-flat")
                gr.HTML("<div class='fq-sect-title'>Error log</div>", elem_classes="fq-flat")
                error_text = gr.Textbox(
                    show_label=False, lines=8, max_lines=20, interactive=False, rtl=False,
                    buttons=["copy"], placeholder="No errors.", elem_id="fq-errlog",
                )

        # ---- settings drawer ----------------------------------------------
        # Layout: (1) the local judge, which is the only AI step on by default;
        # (2) optional LLM features, off by default, each revealing its
        # provider/model only when enabled; (3) retrieval knobs; (4) system info.
        with gr.Sidebar(label="Settings", open=False, position="right", width=420) as sidebar:
            sidebar_header = gr.HTML(
                render_sidebar_header("ar"),
                elem_classes="fq-flat",
            )
            with gr.Accordion("1 · Relevance judge", open=True):
                gr.Markdown(
                    "Checks whether each result **directly answers** your question. "
                    "The default is the fine-tuned Laya model, which runs locally: free, no API key."
                )
                judge_enabled = gr.Checkbox(value=True, label="Judge each result")
                with gr.Group() as judge_opts:
                    judge_provider = gr.Radio(JUDGE_PROVIDERS, value=PROVIDER_LAYA, label="Judge")
                    judge_model = gr.Dropdown(
                        models_for("judge", PROVIDER_LAYA), value=default_model_for("judge", PROVIDER_LAYA),
                        label="Model", allow_custom_value=True, filterable=True,
                    )
                filter_verdicts = gr.CheckboxGroup(
                    choices=[(I18N["ar"]["verdict_answers"], "answers-it"), (I18N["ar"]["verdict_does_not_answer"], "does-not-answer")],
                    value=list(VERDICTS),
                    label="Show results judged as",
                    info=I18N["ar"]["filter_info"],
                )

            with gr.Accordion("2 · Optional AI features (API key needed)", open=False):
                gr.Markdown(
                    "All **off by default**. Each one calls a cloud LLM (Gemini or OpenRouter), "
                    "so it needs an API key and adds a few seconds and a small cost per search."
                )
                rewrite_enabled = gr.Checkbox(
                    value=False,
                    label="Rewrite / translate the question",
                    info="Turns any language or dialect into a clean Arabic search query.",
                )
                with gr.Group(visible=False) as rewrite_opts:
                    rewrite_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                    rewrite_model = gr.Dropdown(
                        models_for("rewrite", PROVIDER_GEMINI), value=default_model_for("rewrite", PROVIDER_GEMINI),
                        label="Model", allow_custom_value=True, filterable=True,
                    )
                consensus_enabled = gr.Checkbox(
                    value=False,
                    label="Scholar consensus",
                    info="Labels whether the scholars agree, partially agree, or disagree.",
                )
                with gr.Group(visible=False) as consensus_opts:
                    consensus_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                    consensus_model = gr.Dropdown(
                        models_for("consensus", PROVIDER_GEMINI), value=default_model_for("consensus", PROVIDER_GEMINI),
                        label="Model", allow_custom_value=True, filterable=True,
                    )
                summary_enabled = gr.Checkbox(
                    value=False,
                    label="Comparative summary",
                    info="Writes an Arabic summary comparing what each scholar said.",
                )
                with gr.Group(visible=False) as summary_opts:
                    summary_provider = gr.Radio(PROVIDERS, value=PROVIDER_GEMINI, label="Provider")
                    summary_model = gr.Dropdown(
                        models_for("summary", PROVIDER_GEMINI), value=default_model_for("summary", PROVIDER_GEMINI),
                        label="Model", allow_custom_value=True, filterable=True,
                    )
                with gr.Accordion("API keys", open=False):
                    gr.Markdown("Leave blank to use `GEMINI_API_KEY` / `OPENROUTER_API_KEY` from your `.env`.")
                    gemini_key_input = gr.Textbox(
                        value="", type="password", label="Gemini API key", placeholder="AIza…",
                    )
                    openrouter_key_input = gr.Textbox(
                        value="", type="password", label="OpenRouter API key", placeholder="sk-or-v1-…",
                    )

            with gr.Accordion("3 · Retrieval", open=False):
                retrieve_k = gr.Slider(5, MAX_RETRIEVE_K, value=DEFAULT_RETRIEVE_K, step=1,
                                       label="Candidates to retrieve", info="Top-K from the embedding search.")
                rerank_pool = gr.Slider(1, DEFAULT_RETRIEVE_K, value=DEFAULT_RERANK_POOL, step=1,
                                        label="Candidates to rerank", info="How many of those the cross-encoder rescores.")
                max_candidate_chars = gr.Slider(500, 8000, value=DEFAULT_MAX_CANDIDATE_CHARS, step=250,
                                                label="Judge text limit", info="Characters of each fatwa sent to the judge.")
                min_rerank_score = gr.Number(value=None, label="Minimum rerank score", info="Leave blank for no filter.")
            with gr.Accordion("4 · System", open=False):
                gpu_desc = "None (CPU only)"
                gpu_vram = "0 MB"
                if torch.cuda.is_available():
                    try:
                        gpu_desc = torch.cuda.get_device_name(0)
                        tot_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        gpu_vram = f"{tot_gb:.1f} GB"
                    except Exception:
                        gpu_desc = "CUDA Device"
                gr.HTML(
                    f"<div class='fq-keys'>{key_chip(GEMINI_API_KEY_ENV, 'Gemini key')}"
                    f"{key_chip(OPENROUTER_API_KEY_ENV, 'OpenRouter key')}</div>"
                    "<dl class='fq-sys' style='margin-top:12px'>"
                    f"<dt>Detected GPU</dt><dd>{esc(gpu_desc)} ({esc(gpu_vram)} VRAM)</dd>"
                    f"<dt>Embedder</dt><dd>{esc(EMBEDDER_MODEL_NAME)} · <b>{esc(emb_dev)}</b></dd>"
                    f"<dt>Reranker</dt><dd><b>ALJIACHI/Mizan-Rerank-v3-Turbo</b> · <b>{esc(rer_dev)}</b></dd>"
                    f"<dt>Judge</dt><dd>{esc(DEFAULT_LAYA_MODEL.split('/')[-1])} · <b>{esc(judge_dev)}</b></dd>"
                    f"<dt>Corpus</dt><dd>{n_fatwas:,} fatwas · {len(available_scholars)} scholars</dd></dl>",
                    elem_classes="fq-flat",
                )
            reset_btn = gr.Button(I18N["ar"]["reset_btn"], size="sm")

        current_search_state = gr.State({})

        # ---- wiring --------------------------------------------------------
        inputs = [
            query,
            search_mode, selected_scholars, x_per_scholar,
            rewrite_enabled, rewrite_provider, rewrite_model,
            judge_enabled, judge_provider, judge_model,
            consensus_enabled, consensus_provider, consensus_model,
            summary_enabled, summary_provider, summary_model,
            retrieve_k, rerank_pool, show_n,
            max_candidate_chars, min_rerank_score, filter_verdicts,
            gemini_key_input, openrouter_key_input,
            lang_state,
        ]
        outputs = [
            query_info, stats_html, retrieval_html, rerank_html, results_html, error_text,
            current_search_state,
        ]

        def search(*args, progress=gr.Progress(track_tqdm=True)):
            query_text = args[0]
            lang = args[-1] if len(args) >= 25 else "ar"
            
            if not (query_text or "").strip():
                yield (
                    I18N[lang]["empty_query"],
                    EMPTY_PIPELINE, EMPTY_TABLE, EMPTY_TABLE, render_empty_results(lang), "",
                    {},
                )
                return

            progress(0.05, desc="Analyzing question..." if lang == "en" else "جاري تحليل السؤال...")
            yield (
                I18N[lang]["loading_query"], EMPTY_PIPELINE, EMPTY_TABLE, EMPTY_TABLE, render_loading_results(lang), "",
                {},
            )

            def progress_callback(pct, msg):
                progress(pct, desc=msg)

            search_args = list(args[:-1])
            q_info, stats, retrieved, reranked, cards, err, display_rows, judge_map, search_query = app.run_search(
                *search_args,
                lang=lang,
                progress_cb=progress_callback,
            )

            cands_dict = {}
            if display_rows is not None and len(display_rows):
                for idx, (_, row) in enumerate(display_rows.iterrows()):
                    fid = int(row.fatwa_id)
                    j_obj = judge_map.get(fid) if judge_map else None
                    cands_dict[fid] = {
                        "source": row.source,
                        "title": row.title,
                        "answer": row.answer,
                        "reranker_score": float(row.get("reranker_score", 0.0)),
                        "laya_verdict": j_obj.verdict if j_obj else None,
                        "laya_prob": (j_obj.probabilities.get("answers-it") if (j_obj and j_obj.probabilities) else None),
                    }
            new_state = {
                "query": query_text,
                "search_query": search_query,
                "candidates": cands_dict,
            }

            yield (
                q_info if str(q_info).lstrip().startswith("<") else f'<div class="fq-note">{esc(q_info)}</div>',
                stats,
                df_to_html(retrieved),
                df_to_html(reranked),
                cards or f'<p class="empty">{I18N[lang]["empty_failed"]}</p>',
                err,
                new_state,
            )

        run_kw = dict(fn=search, inputs=inputs, outputs=outputs, show_progress="full")
        query.submit(**run_kw)

        def handle_modal_feedback(search_state: dict[str, Any], payload_str: str) -> None:
            if not payload_str or not search_state:
                return
            try:
                data = json.loads(payload_str)
                fid = data.get("fid")
                ftype = data.get("type", "up")
                comment = data.get("comment", "")
                record_user_feedback(app.s.data_dir, search_state, str(fid), ftype, comment)
            except Exception as e:
                print(f"Error handling modal feedback: {e}")

        fb_trigger.click(
            fn=handle_modal_feedback,
            inputs=[current_search_state, fb_payload],
            outputs=[],
            show_progress="hidden",
        )

        theme_btn.click(
            fn=None,
            inputs=[],
            outputs=[],
            js="""() => {
                const isDarkNow = !document.body.classList.contains('dark');
                document.body.classList.toggle('dark', isDarkNow);
                document.documentElement.classList.toggle('dark', isDarkNow);
                const gApp = document.querySelector('gradio-app');
                if (gApp) gApp.classList.toggle('dark', isDarkNow);
                try { localStorage.setItem('fq-theme-mode', isDarkNow ? 'dark' : 'light'); } catch(e) {}
                const btn = document.getElementById('fq-theme-btn');
                if (btn) {
                    const b = btn.tagName === 'BUTTON' ? btn : (btn.querySelector('button') || btn);
                    b.innerText = isDarkNow ? "Light" : "Dark";
                }
            }""",
        )

        def toggle_language(current_l: str, current_q_info: str, current_res: str):
            new_l = "en" if current_l == "ar" else "ar"
            q_info_up = gr.update()
            if not current_q_info or "fq-empty-sm" in current_q_info:
                q_info_up = gr.update(value=I18N[new_l]["empty_query"])
            res_up = gr.update()
            if not current_res or "fq-empty-title" in current_res:
                res_up = gr.update(value=render_empty_results(new_l))

            return (
                new_l,
                gr.update(value=I18N[new_l]["lang_btn"]),
                gr.update(value=render_brand_html(new_l, n_fatwas, len(available_scholars))),
                gr.update(placeholder=I18N[new_l]["placeholder"], rtl=(new_l == "ar")),
                gr.update(
                    label=I18N[new_l]["search_mode_label"],
                    choices=SEARCH_MODES if new_l == "ar" else SEARCH_MODES_EN,
                ),
                gr.update(label=I18N[new_l]["x_per_scholar_label"]),
                gr.update(label=I18N[new_l]["show_n_label"]),
                gr.update(label=I18N[new_l]["scholars_label"]),
                gr.update(value=I18N[new_l]["settings_btn"]),
                gr.update(
                    label=I18N[new_l]["filter_title"],
                    info=I18N[new_l]["filter_info"],
                    choices=[(I18N[new_l]["verdict_answers"], "answers-it"), (I18N[new_l]["verdict_does_not_answer"], "does-not-answer")],
                ),
                gr.update(value=render_sidebar_header(new_l)),
                q_info_up,
                res_up,
            )

        lang_btn.click(
            fn=toggle_language,
            inputs=[lang_state, query_info, results_html],
            outputs=[
                lang_state,
                lang_btn,
                brand_html,
                query,
                search_mode,
                x_per_scholar,
                show_n,
                selected_scholars,
                settings_btn,
                filter_verdicts,
                sidebar_header,
                query_info,
                results_html,
            ],
            show_progress="hidden",
            js="""() => {
                window._currentLang = (window._currentLang === 'en') ? 'ar' : 'en';
            }""",
        )

        settings_btn.click(
            lambda is_open: (not is_open, gr.Sidebar(open=not is_open)),
            inputs=sidebar_open, outputs=[sidebar_open, sidebar], show_progress="hidden",
        )
        sidebar.expand(lambda: True, outputs=sidebar_open, show_progress="hidden")
        sidebar.collapse(lambda: False, outputs=sidebar_open, show_progress="hidden")

        # Toggle dynamic controls based on search mode
        search_mode.change(
            lambda mode: (
                gr.update(visible=(mode == SEARCH_MODE_PER_SCHOLAR)),
                gr.update(visible=(mode == SEARCH_MODE_OVERALL)),
            ),
            inputs=search_mode,
            outputs=[x_per_scholar, show_n],
            show_progress="hidden",
        )

        # Dependent controls
        rewrite_enabled.change(
            lambda e: gr.update(visible=e),
            inputs=rewrite_enabled, outputs=rewrite_opts, show_progress="hidden",
        )
        judge_enabled.change(
            lambda e: gr.update(visible=e),
            inputs=judge_enabled, outputs=judge_opts, show_progress="hidden",
        )
        consensus_enabled.change(
            lambda e: gr.update(visible=e),
            inputs=consensus_enabled, outputs=consensus_opts, show_progress="hidden",
        )
        summary_enabled.change(
            lambda e: gr.update(visible=e),
            inputs=summary_enabled, outputs=summary_opts, show_progress="hidden",
        )

        rewrite_provider.change(
            lambda p: gr.update(choices=models_for("rewrite", p), value=default_model_for("rewrite", p)),
            inputs=rewrite_provider, outputs=rewrite_model,
        )
        judge_provider.change(
            lambda p: gr.update(choices=models_for("judge", p), value=default_model_for("judge", p)),
            inputs=judge_provider, outputs=judge_model,
        )
        consensus_provider.change(
            lambda p: gr.update(choices=models_for("consensus", p), value=default_model_for("consensus", p)),
            inputs=consensus_provider, outputs=consensus_model,
        )
        summary_provider.change(
            lambda p: gr.update(choices=models_for("summary", p), value=default_model_for("summary", p)),
            inputs=summary_provider, outputs=summary_model,
        )

        retrieve_k.change(
            lambda k, rp: gr.update(maximum=int(k), value=min(int(rp), int(k))),
            inputs=[retrieve_k, rerank_pool], outputs=rerank_pool,
        )
        rerank_pool.change(
            lambda rp, sn: gr.update(maximum=int(rp), value=min(int(sn), int(rp))),
            inputs=[rerank_pool, show_n], outputs=show_n,
        )

        def reset_settings(lang: str = "ar"):
            return (
                False, PROVIDER_GEMINI, default_model_for("rewrite", PROVIDER_GEMINI),
                True, PROVIDER_LAYA, default_model_for("judge", PROVIDER_LAYA),
                False, PROVIDER_GEMINI, default_model_for("consensus", PROVIDER_GEMINI),
                False, PROVIDER_GEMINI, default_model_for("summary", PROVIDER_GEMINI),
                DEFAULT_RETRIEVE_K, DEFAULT_RERANK_POOL, DEFAULT_SHOW_N,
                DEFAULT_MAX_CANDIDATE_CHARS, None, list(VERDICTS),
            )

        reset_btn.click(
            reset_settings,
            inputs=[lang_state],
            outputs=[
                rewrite_enabled, rewrite_provider, rewrite_model,
                judge_enabled, judge_provider, judge_model,
                consensus_enabled, consensus_provider, consensus_model,
                summary_enabled, summary_provider, summary_model,
                retrieve_k, rerank_pool, show_n,
                max_candidate_chars, min_rerank_score, filter_verdicts,
            ],
            show_progress="hidden",
        )

    return demo


# ---------------------------------------------------------------------------
# CLI & Entrypoint
# ---------------------------------------------------------------------------


def ensure_data(data_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if multi-scholar assets are already in place
    has_multi = (data_dir / "multi_scholar_clean.csv").exists() and (
        data_dir / "qwen3_06b_multi_scholar_embeddings.pt"
    ).exists()
    if has_multi:
        return

    # Check if Bin Baz assets are already in place
    has_binbaz = (data_dir / "binbaz_clean.csv").exists() and (
        data_dir / "qwen3_06b_title_answer_embeddings.pt"
    ).exists()

    # Try downloading multi-scholar files from DATA_REPO_ID first
    print(f"Checking for multi-scholar assets in Hugging Face dataset '{DATA_REPO_ID}'...")
    multi_success = True
    for name in MULTI_SCHOLAR_DATA_FILES:
        target = data_dir / name
        if target.exists():
            continue
        try:
            print(f"Downloading {name} from {DATA_REPO_ID} ...")
            hf_hub_download(repo_id=DATA_REPO_ID, repo_type="dataset", filename=name, local_dir=data_dir)
        except Exception as e:
            print(f"Notice: Could not download {name} from {DATA_REPO_ID} ({type(e).__name__}).")
            multi_success = False
            break

    if multi_success and (data_dir / "multi_scholar_clean.csv").exists():
        print("Successfully fetched multi-scholar assets from Hugging Face.")
        return

    # If multi-scholar files were not found on HF repo and not already on disk, fall back to Bin Baz
    if not has_binbaz:
        print(f"Falling back to Bin Baz base dataset in '{DATA_REPO_ID}'...")
        for name in BINBAZ_DATA_FILES:
            if (data_dir / name).exists():
                continue
            try:
                print(f"Downloading {name} from {DATA_REPO_ID} ...")
                hf_hub_download(repo_id=DATA_REPO_ID, repo_type="dataset", filename=name, local_dir=data_dir)
            except Exception as e:
                print(f"Warning: Failed to download {name}: {e}")


def default_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arabic Fatwa Q&A: semantic search and reranking over trusted Islamic fatwas."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FATWA_DATA_DIR", APP_DIR / "data")),
        help="Folder with cleaned CSV + embeddings.pt + metadata.json.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not fetch missing data files from Hugging Face.")
    parser.add_argument("--embedder-device", choices=["cpu", "cuda", "auto"], default="cpu", help="Embedder device.")
    parser.add_argument("--reranker-device", choices=["auto", "cuda", "cpu"], default="auto", help="Reranker device.")
    parser.add_argument("--judge-device", choices=["auto", "cuda", "cpu"], default="auto", help="Judge device for local Laya model.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip the startup warmup query.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port (default 7860).")
    parser.add_argument("--share", action="store_true", help="Create a temporary public *.gradio.live link.")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser tab.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.no_download:
        ensure_data(args.data_dir)

    settings = Settings(
        data_dir=args.data_dir,
        device_embedder=default_device(args.embedder_device),
        device_reranker=default_device(args.reranker_device),
        device_judge=default_device(args.judge_device),
        warmup=not args.no_warmup,
    )

    app = FatwaApp(settings)
    demo = build_ui(app)

    url = f"http://{args.host}:{args.port}"
    print(f"\nFatwa Search running on {url}")
    if not args.no_browser and not args.share:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    launch_kw = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": False,
    }
    # Pass theme and css to launch if supported by Gradio version
    try:
        demo.launch(**launch_kw, theme=THEME, css=CSS)
    except TypeError:
        demo.launch(**launch_kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
