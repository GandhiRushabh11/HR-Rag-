import io
import json
import logging

from docx import Document as DocxDocument
from google.cloud import storage
from pptx import Presentation
from pypdf import PdfReader

from hr_assistant import config
from hr_assistant.document_loader import extract_policy_category

logger = logging.getLogger(__name__)

_RAW_TO_PROCESSED = {
    config.GCS_PREFIX: config.PROCESSED_HR_PREFIX,
    config.GCS_NOISE_PREFIX: config.PROCESSED_NOISE_PREFIX,
}


def _parse_pdf(raw_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(raw_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text


def _parse_docx(raw_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    doc = DocxDocument(io.BytesIO(raw_bytes))
    text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text


def _parse_pptx(raw_bytes: bytes) -> str:
    """Extract text from a PPTX file."""
    presentation = Presentation(io.BytesIO(raw_bytes))
    lines = []
    for slide in presentation.slides:
        if slide.shapes.title is not None:
            lines.append(slide.shapes.title.text)
        for shape in slide.shapes:
            if shape.has_text_frame and shape != slide.shapes.title:
                lines.append(shape.text_frame.text)
    text = "\n".join(lines)
    return text


_PARSERS = {".pdf": _parse_pdf, ".docx": _parse_docx, ".pptx": _parse_pptx}


def parse_blob(blob) -> str:
    """Download a blob and return its plain text, dispatching on extension.
    .txt decodes directly; other formats go through the matching parser."""
    ext = "." + blob.name.rsplit(".", 1)[-1].lower() if "." in blob.name else ""
    if ext == ".txt":
        return blob.download_as_text()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"No parser registered for file extension {ext!r} ({blob.name})")
    return parser(blob.download_as_bytes())


def process_raw_to_json(bucket_name: str = config.GCS_BUCKET_NAME) -> int:
    """Parse every raw file (any format) and write it as a processed JSON
    record in GCS. Returns how many records were written.

    The record holds exactly what load_processed_documents_from_gcs() reads
    back: source, policy_category, text, raw_gcs_path.
    """

    client = storage.Client(project=config.PROJECT_ID)
    bucket = client.bucket(bucket_name)

    count = 0
    for raw_prefix, processed_prefix in _RAW_TO_PROCESSED.items():
        for blob in bucket.list_blobs(prefix=raw_prefix):
            filename = blob.name.rsplit("/", 1)[-1]
            if "." not in filename:
                continue  # skip folders
            try:
                text = parse_blob(blob)
            except Exception as e:
                logger.error(f"Failed to parse {blob.name}: {e}")
                continue
            record = {
                "source": blob.name.rsplit("/", 1)[-1],
                "policy_category": extract_policy_category(text),
                "text": text,
                "raw_gcs_path": f"gs://{bucket_name}/{blob.name}",
            }
            processed_blob_name = filename.rsplit(".", 1)[0] + ".json"
            processed_blob = bucket.blob(f"{processed_prefix}{processed_blob_name}")
            processed_blob.upload_from_string(
                json.dumps(record, indent=2), content_type="application/json"
            )
            logger.info(f" %s -> %s", blob.name, processed_blob.name)
            count += 1
    return count
