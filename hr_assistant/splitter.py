from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config


def split_into_chunks(documents):
    """Split documents into small overlapping chunks. Metadata (source,
    policy_category) carries over onto every chunk automatically."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
