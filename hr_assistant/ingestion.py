import glob
import logging
import os

from google.cloud import storage

from hr_assistant import config
from hr_assistant.document_loader import (
    load_processed_documents_from_gcs,
    load_documents_from_gcs,
)

from hr_assistant.processor import process_raw_to_json
from hr_assistant.splitter import split_into_chunks
from hr_assistant.vector_store import build_vector_store, collection_exists

logger = logging.getLogger(__name__)

__LOCAL_HR_GLOB = os.path.join("data", "*.txt")
__LOCAL_NOISE_GLOB = os.path.join("data", "noise", "*")


def upload_corpus_to_gcs() -> None:
    """Upload every local .txt file under data/ and data/noise/ to GCS
    under the configured prefixes. This is a one-time operation for
    initial ingestion; after that, new files can be added directly to GCS.
    """
    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(config.GCS_BUCKET_NAME)

    pairs = [
        (__LOCAL_HR_GLOB, config.GCS_PREFIX),
        (__LOCAL_NOISE_GLOB, config.GCS_NOISE_PREFIX),
    ]
    for local_glob, gcs_prefix in pairs:
        files = [f for f in glob.glob(local_glob) if os.path.isfile(f)]
        for path in files:
            blob = bucket.blob(f"{gcs_prefix}/{os.path.basename(path)}")
            blob.upload_from_filename(path)
        logger.info(
            "Uploaded %d file(s) -> gs://%s/%s",
            len(files),
            config.GCS_BUCKET_NAME,
            gcs_prefix,
        )


def ingest_hr_policies(force: bool = False) -> None:
    """Ingest HR policies and noise docs into Qdrant Cloud.

    Steps:
    1. Upload local .txt files to GCS (one-time; skip if already done)
    2. Parse raw files into JSON records in the processed zone
    3. Load processed JSON records as LangChain Documents
    4. Split Documents into chunks
    5. Embed chunks and upsert into Qdrant collection(s)

    If force=True, re-ingest even if the Qdrant collection already exists.
    """

    if not force and collection_exists(config.QDRANT_COLLECTION_NAME):
        logger.info(
            "Qdrant collection %s already exists; skipping ingestion",
            config.QDRANT_COLLECTION_NAME,
        )
        return

    documents = load_documents_from_gcs()
    chunks = split_into_chunks(documents)
    build_vector_store(
        chunks, hybrid=True, collection_name=config.QDRANT_COLLECTION_NAME
    )
    logger.info(
        "Ingested %d HR policies into Qdrant collection %s",
        len(documents),
        config.QDRANT_COLLECTION_NAME,
    )


def ingest_noisy_corpus(force: bool = False) -> None:
    """Ingest HR policies and non-HR noise docs into Qdrant Cloud.

    Steps:
    1. Upload local .txt files to GCS (one-time; skip if already done)
    2. Parse raw files into JSON records in the processed zone
    3. Load processed JSON records as LangChain Documents
    4. Split Documents into chunks
    5. Embed chunks and upsert into Qdrant collection(s)

    If force=True, re-ingest even if the Qdrant collection already exists.
    """

    if not force and collection_exists(config.QDRANT_NOISY_COLLECTION_NAME):
        logger.info(
            "Qdrant collection %s already exists; skipping ingestion",
            config.QDRANT_NOISY_COLLECTION_NAME,
        )
        return

    if force:
        logger.info("Parsing raw files into the processed/ zone...")
        process_raw_to_json()

    documents = load_processed_documents_from_gcs()
    if not documents:
        logger.info("Processed zone is empty — parsing raw files...")
        process_raw_to_json()
        documents = load_processed_documents_from_gcs()

    if not documents:
        raise RuntimeError(
            "Still no processed documents after parsing — check the GCS raw zone."
        )

    chunks = split_into_chunks(documents)
    build_vector_store(
        chunks,
        hybrid=True,
        collection_name=config.QDRANT_NOISY_COLLECTION_NAME,
    )
    logger.info(
        "Ingested %d document(s) (HR + noise) -> %d chunk(s) into '%s'.",
        len(documents),
        len(chunks),
        config.QDRANT_NOISY_COLLECTION_NAME,
    )


def run_ingestion(
    force: bool = False, hr: bool = True, noisy: bool = True, upload: bool = True
) -> None:
    config.check_api_keys()
    if upload:
        upload_corpus_to_gcs()
    if hr:
        ingest_hr_policies(force=force)
    if noisy:
        ingest_noisy_corpus(force=force)
