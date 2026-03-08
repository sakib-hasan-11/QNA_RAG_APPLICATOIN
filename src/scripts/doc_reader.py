import os
import sys

from langchain_community.document_loaders import PyPDFDirectoryLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)


def read_doc(dir):
    """
    Read PDF documents from a directory.

    Args:
        dir (str): Directory path containing PDF files

    Returns:
        list: List of loaded documents
    """
    file_loader = PyPDFDirectoryLoader(dir)

    try:
        docs = file_loader.load()
        logger.info(f"Successfully loaded {len(docs)} documents from {dir}")
        return docs

    except FileNotFoundError as e:
        logger.error(f"PDF file path not correct.... PDF file not found at {dir}")
        raise
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise
