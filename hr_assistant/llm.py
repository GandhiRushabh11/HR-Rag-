from hr_assistant import config
import litellm
from langchain_litellm import ChatLiteLLMRouter
from litellm import Router

# Gemini accepts params (e.g. some tool/response-format options) that Groq's
# OpenAI-compatible endpoint rejects. Drop unsupported params on the way to
# whichever backend serves the call, rather than 400-ing on the fallback.
litellm.drop_params = True

_PRIMARY_GROUP = "hr-llm"
_FALLBACK_GROUP = "hr-llm-fallback"

# One Router for the whole process — it holds per-deployment state
# (cooldowns, retry counters). get_llm() is called per guardrail check and
# once per agent build, so a fresh Router each call would throw that away.
_router: "Router | None" = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(
            model_list=[
                {
                    "model_name": _PRIMARY_GROUP,
                    "litellm_params": {
                        "model": f"gemini/{config.LLM_MODEL_NAME}",
                        "api_key": config.GOOGLE_API_KEY,
                    },
                },
                {
                    "model_name": _FALLBACK_GROUP,
                    "litellm_params": {
                        "model": f"groq/{config.FALLBACK_MODEL_NAME}",
                        "api_key": config.GROQ_API_KEY,
                    },
                },
            ],
            # Keep provider failures bounded, then cross over to Groq.
            timeout=60,
            num_retries=1,
            fallbacks=[{_PRIMARY_GROUP: [_FALLBACK_GROUP]}],
        )
    return _router


def get_llm():
    """The app's chat model: Gemini primary, Groq fallback, one retry policy.

    A LangChain chat model (supports tool calling and structured output), so
    agent.py and guardrails.py use it unchanged. temperature=0 is set once
    here — the single source of truth for both backends."""
    return ChatLiteLLMRouter(
        router=_get_router(),
        model_name=_PRIMARY_GROUP,
        temperature=config.LLM_TEMPERATURE,
    )
