"""01 · config — every setting, read from .env. Everything imports this.

config.py holds values only (keys, URLs, model ids, sizes). The
system-prompt text lives next door in prompts.py (02).
"""

import os
from dotenv import load_dotenv

load_dotenv()

## GCP

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")  # Vertex AI region — used by hr_assistant/llm.py

## ENV VAR / SECRETS

JINA_API_KEY = os.getenv("JINA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

## CLOUD STORAGE

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Raw zone — original files, untouched. Ingestion uploads the local data/
# tree here: HR policies under GCS_PREFIX, the non-HR noise docs under
# GCS_NOISE_PREFIX.
GCS_PREFIX = "raw/hr-policies/"
GCS_NOISE_PREFIX = "raw/other-data/"

# Processed zone — one JSON record per raw file (parsed text + metadata),
# written by processor.py (05). Ingestion reads ONLY from here, so
# PDFs/DOCX/PPTX are parsed once, not on every rebuild.
PROCESSED_HR_PREFIX = "processed/hr-policies/"
PROCESSED_NOISE_PREFIX = "processed/other-data/"

## QDRANT

QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hr_policies")

# A second collection holding HR docs + non-HR noise together. `ingest.py`
# builds it alongside the clean one; the app connects to the clean
# collection. The noisy collection is where a later stage tests retrieval
# against real cross-domain noise.
QDRANT_NOISY_COLLECTION_NAME = os.getenv(
    "QDRANT_NOISY_COLLECTION_NAME", "hr_policies_noisy"
)

## MODELS
# All three are env-overridable so a model swap needs no code change.

# gemini-2.5-flash retires ~2026-10-20 (verified Sept 2026) — set
# LLM_MODEL_NAME to a Gemini 3.x Flash model before then. Current IDs:
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))  # 0=deterministic


# Fallback model — served by Groq if the Vertex Gemini call errors (see
# hr_assistant/llm.py's LiteLLM Router). llm.py prefixes this with "groq/".
# Needs GROQ_API_KEY set; without it the primary still works, the fallback
# just can't fire.
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "openai/gpt-oss-20b")


# jina-embeddings-v2-base-en: 768-dim, English. Changing it changes the
# vector dimension, so it also needs `python ingest.py --force`.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jina-embeddings-v2-base-en")

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME", "jina-reranker-v2-base-multilingual"
)

## CHUNK / TEXT SPLITTING

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

## RETRIEVAL RESULTS

# Broad questions ("list all maternity leave provisions") need enough
# chunks in context to answer in full; at CHUNK_SIZE=500 a single
# multi-section policy doc is often 4-5 chunks, so a small top_k can't
# return the whole thing no matter how the prompt is worded.
RETRIEVER_TOP_K = os.getenv(
    "RETRIEVER_TOP_K", 5
)  # how many chunks to retrieve for LLM context
RERANK_CANDIDATE_K = 12  # wider shortlist retrieved before re-ranking
# Exact values used in each policy file's "Policy Category:" line. The
# search tool hard-filters retrieval to only these categories — non-HR
# chunks (Finance/Sales/Operations/Business) are structurally unreachable,
# regardless of how confused embedding similarity gets.
HR_POLICY_CATEGORIES = {
    "Leave",
    "Work From Home",
    "Probation",
    "Notice Period",
    "Reimbursement",
    "Code of Conduct",
    "Holidays",
    "Maternity Paternity",
    "Travel Expense",
    "Exit Process",
}

# Post-rerank relevance score cutoff — below this, the tool reports "not
# found" instead of returning a stray, technically-in-scope-but-irrelevant
# chunk. Jina relevance scores are ~0-1; this is a starting point, calibrate
# empirically against real queries rather than trusting it blindly.
RELEVANCE_THRESHOLD = os.getenv("RELEVANCE_THRESHOLD", "0.35")

## RELIABILITY — input/output safety guardrail (Model Armor)

GUARDRAIL_PROVIDER = os.getenv(
    "GUARDRAIL_PROVIDER", "model_armor"
)  # "model_armor" | "gemini_lite" | "none"
MODEL_ARMOR_LOCATION = os.getenv(
    "MODEL_ARMOR_LOCATION", "us"
)  # multi-region; verify supported regions at setup time
MODEL_ARMOR_TEMPLATE_ID = os.getenv("MODEL_ARMOR_TEMPLATE_ID", "hr-assistant-guardrail")

# What to do when the guardrail PROVIDER itself errors (an API failure, not
# a content block). Input fails closed — an unscreened prompt must never
# reach the model. Output fails open — a transient screening error
# shouldn't discard an answer the model already produced. Both overridable.
# See hr_assistant/guardrails.py.
GUARDRAIL_FAIL_OPEN_INPUT = (
    os.getenv("GUARDRAIL_FAIL_OPEN_INPUT", "false").strip().lower() == "true"
)
GUARDRAIL_FAIL_OPEN_OUTPUT = (
    os.getenv("GUARDRAIL_FAIL_OPEN_OUTPUT", "true").strip().lower() == "true"
)

# The input guardrail screens the current question plus up to this many of
# the most recent *user* turns for the thread (0 disables history
# screening) — so a multi-turn attack that looks harmless message-by-message
# is still caught in aggregate. Assistant answers and tool output are never
# included. See hr_assistant/thread_memory.py's input_text_for_screening.
GUARDRAIL_HISTORY_TURNS = int(os.getenv("GUARDRAIL_HISTORY_TURNS", "6"))

## RELIABILITY — semantic cache

SEMANTIC_CACHE_THRESHOLD = int(
    os.getenv("SEMANTIC_CACHE_THRESHOLD", 0.93)
)  # cosine similarity above this = cache hit

# Bound the in-memory cache: a long-lived process shouldn't grow forever,
# and a stale answer (policy changed + re-ingested) shouldn't be served
# indefinitely. Past MAX_ENTRIES the oldest entry is evicted; entries older
# than TTL_SECONDS are ignored on lookup (0 = never expire).
SEMANTIC_CACHE_MAX_ENTRIES = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "500"))
SEMANTIC_CACHE_TTL_SECONDS = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "3600"))

## LANGSMITH — request tracing (env-var based; langchain auto-traces)

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "hr-policy-assistant")

## JUDGE LLM — Groq, read by hr_assistant/evaluation.py.
# A different model family from the app's Gemini, so the eval isn't the
# model grading its own answers. gpt-oss on Groq is OpenAI-API-compatible,
# so langchain-openai's ChatOpenAI talks to it directly — no new dependency.
# GROQ_API_KEY is also the credential for the app's fallback model
# (FALLBACK_MODEL_NAME, above — see hr_assistant/llm.py).

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "openai/gpt-oss-120b")


def check_api_keys() -> None:
    """Stop early with a clear message if a required key/config is missing."""
    missing = [
        name
        for name, value in [
            ("PROJECT_ID", PROJECT_ID),
            ("JINA_API_KEY", JINA_API_KEY),
            ("QDRANT_URL", QDRANT_URL),
            ("QDRANT_API_KEY", QDRANT_API_KEY),
            ("GCS_BUCKET_NAME", GCS_BUCKET_NAME),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required .env values: {', '.join(missing)}")
