import uuid
from functools import lru_cache

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langsmith import client
from qdrant_client import QdrantClient, hybrid
from qdrant_client.models import FieldCondition, Filter, MatchAny, PayloadSchemaType

# langchain doc
# page contetn and metadata are stored in the payload,
# and the vector is stored in the vector field.

from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_model

_ID_NAMESPACE = uuid.UUID("5b9c1a1e-8b1a-4f7d-9d5e-2b6a7c9d1e3f")  # fixed, arbitrary
_SPARSE_MODEL = "Qdrant/bm25"  # keyword search


@lru_cache(maxsize=1)
def _qdrant_client() -> QdrantClient:
    """Return a cached QdrantClient instance for the configured host."""
    return QdrantClient(
        url=config.QDRANT_URL,
        prefer_grpc=True,
        api_key=config.QDRANT_API_KEY,
    )


def collection_exists(collection_name: str) -> bool:
    """Return True if the given collection exists in Qdrant."""
    client = _qdrant_client()
    if not client.collection_cluster_info(collection_name):
        return False
    return client.count(collection_name).count > 0


def load_vector_store(
    collection_name: str = config.QDRANT_COLLECTION_NAME, hybrid: bool = True
) -> QdrantVectorStore:
    """Return a LangChain QdrantVectorStore for the given collection."""

    if not collection_exists(collection_name):
        raise ValueError(f"Qdrant collection {collection_name} does not exist")

    kwargs = dict(
        collection_name=collection_name,
        embedding=get_embeddings_model(),
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    if hybrid:
        kwargs["sparse_embedding"] = FastEmbedSparse(model_name=_SPARSE_MODEL)
        kwargs["retrieval_mode"] = RetrievalMode.HYBRID
    else:
        kwargs["retrieval_mode"] = RetrievalMode.DENSE

    return QdrantVectorStore.from_existing_collection(**kwargs)


def _stable_chunk_id(chunk) -> str:
    """Return a stable UUID for a given text chunk, based on its content and index."""
    key = f"{chunk.metadata.get('source', '')}::{chunk.page_content}"
    return str(uuid.uuid5(_ID_NAMESPACE, key))


def build_vector_store(
    chunks, hybrid: bool = True, collection_name: str = config.QDRANT_COLLECTION_NAME
) -> QdrantVectorStore:
    """Embed every chunk and upsert into a Qdrant Cloud collection.

    hybrid=True (the default, matching load_vector_store) also computes
    sparse (keyword/BM25) vectors — a DENSE-only collection built here
    can't later be opened in hybrid mode. collection_name defaults to the
    clean HR collection; ingestion also builds the mixed
    config.QDRANT_NOISY_COLLECTION_NAME.
    """

    kwargs = dict(
        documents=chunks,
        embedding=get_embeddings_model(),
        ids=[_stable_chunk_id(chunk) for chunk in chunks],
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=collection_name,
        force_recreate=False,  # always rebuild the collection from scratch
    )

    if hybrid:
        kwargs["sparse_embedding"] = FastEmbedSparse(model_name=_SPARSE_MODEL)
        kwargs["retrieval_mode"] = RetrievalMode.HYBRID
    else:
        kwargs["retrieval_mode"] = RetrievalMode.DENSE

    store = QdrantVectorStore.from_documents(**kwargs)

    # Create a payload index for the policy_category field to enable filtering by category.
    store.client.create_payload_index(
        collection_name=collection_name,
        field_name="policy_category",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    return store


def get_retriever(
    vector_store: QdrantVectorStore,
    top_k: int = config.RETRIEVER_TOP_K,
    filter_categories: set[str] | list[str] | None = None,
):
    """Return a retriever for the given vector store, optionally filtered by policy_category."""

    search_kwargs = {"k": top_k}
    if filter_categories:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.policy_category",
                    match=MatchAny(any=filter_categories),
                )
            ]
        )
    return vector_store.as_retriever(**search_kwargs)
