from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config


def get_embeddings_model() -> JinaEmbeddings:
    """Return a JinaEmbeddings instance for the configured model name."""
    return JinaEmbeddings(
        jina_api_key=config.JINA_API_KEY, model_name=config.EMBEDDING_MODEL_NAME
    )
