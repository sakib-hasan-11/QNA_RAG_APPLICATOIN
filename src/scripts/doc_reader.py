import os
import sys
import tempfile
from pathlib import Path

import boto3
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)

# S3 Configuration - Set your S3 bucket and file path here
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "your-bucket-name")
S3_FILE_KEY = os.getenv("S3_FILE_KEY", "path/to/your/document.pdf")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def download_from_s3(bucket_name, file_key, local_dir):
    """
    Download a PDF file from S3 bucket to local directory.

    Args:
        bucket_name (str): S3 bucket name
        file_key (str): S3 object key (file path in bucket)
        local_dir (str): Local directory to save the file

    Returns:
        str: Path to the downloaded file
    """
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)

        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Extract filename from S3 key
        filename = os.path.basename(file_key)
        local_file_path = os.path.join(local_dir, filename)

        logger.info(f"Downloading {file_key} from S3 bucket {bucket_name}...")
        s3_client.download_file(bucket_name, file_key, local_file_path)
        logger.info(f"✓ Successfully downloaded to {local_file_path}")

        return local_file_path

    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        raise


def read_doc(dir=None, use_s3=False):
    """
    Read PDF documents from a directory or S3 bucket.

    Args:
        dir (str): Directory path containing PDF files (for local files)
        use_s3 (bool): If True, download from S3 bucket instead

    Returns:
        list: List of loaded documents
    """
    try:
        if use_s3:
            logger.info("Loading documents from S3 bucket")
            # Create temporary directory for S3 downloads
            temp_dir = tempfile.mkdtemp()

            # Download file from S3
            local_file_path = download_from_s3(S3_BUCKET_NAME, S3_FILE_KEY, temp_dir)

            # Load the PDF
            file_loader = PyPDFLoader(local_file_path)
            docs = file_loader.load()

        else:
            # Load from local directory
            logger.info(f"Loading documents from local directory: {dir}")
            file_loader = PyPDFDirectoryLoader(dir)
            docs = file_loader.load()

        logger.info(f"✓ Successfully loaded {len(docs)} documents")
        return docs

    except FileNotFoundError as e:
        logger.error(f"PDF file path not correct.... PDF file not found")
        raise
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise
