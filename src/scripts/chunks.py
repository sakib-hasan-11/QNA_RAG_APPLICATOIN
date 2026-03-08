from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)

def divide_chunks(docs, chunk_size=800, chunk_overlap=80):
    """
    Divide documents into smaller chunks.
    
    Args:
        docs (list): List of documents to chunk
        chunk_size (int): Size of each chunk (default: 800)
        chunk_overlap (int): Overlap between chunks (default: 80)
    
    Returns:
        list: List of chunked documents
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size
        )

        docs_chunk = text_splitter.split_documents(docs)

        logger.info(f'Chunks created successfully: {len(docs_chunk)} chunks from {len(docs)} documents')

        return docs_chunk
    except Exception as e:
        logger.error(f'Error creating chunks: {e}')
        raise