from langchain_google_genai import ChatGoogleGenerativeAI

from hr_assistant import config


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a ChatGoogleGenerativeAI instance for the configured model name."""
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
        request_timeout=60,
        retries=1,
    )
