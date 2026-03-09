"""
Pipeline Runner for ECS
This script runs the complete indexing pipeline to load documents from S3
and store vectors in Pinecone.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from pipelines.pipeline import RAGPipeline
from utils.log import get_logger

logger = get_logger(__name__)


def main():
    """
    Run the complete indexing pipeline.
    Environment variables required:
    - OPENAI_API_KEY: OpenAI API key
    - PINECONE_API_KEY: Pinecone API key
    - PINECONE_HOST: Pinecone host URL
    - S3_BUCKET_NAME: S3 bucket name
    - S3_FILE_KEY: S3 file key (path to PDF in bucket)
    - AWS_REGION: AWS region (optional, default: us-east-1)
    """
    try:
        logger.info("=" * 80)
        logger.info("STARTING RAG PIPELINE IN ECS")
        logger.info("=" * 80)

        # Validate required environment variables
        required_env_vars = [
            "OPENAI_API_KEY",
            "PINECONE_API_KEY",
            "PINECONE_HOST",
            "S3_BUCKET_NAME",
            "S3_FILE_KEY",
        ]

        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            sys.exit(1)

        # Initialize pipeline
        pipeline = RAGPipeline()

        # Configuration
        embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "80"))
        batch_size = int(os.getenv("BATCH_SIZE", "100"))

        logger.info(f"Configuration:")
        logger.info(f"  - Embedding Model: {embed_model}")
        logger.info(f"  - LLM Model: {llm_model}")
        logger.info(f"  - Chunk Size: {chunk_size}")
        logger.info(f"  - Chunk Overlap: {chunk_overlap}")
        logger.info(f"  - Batch Size: {batch_size}")
        logger.info(f"  - S3 Bucket: {os.getenv('S3_BUCKET_NAME')}")
        logger.info(f"  - S3 File: {os.getenv('S3_FILE_KEY')}")

        # Run indexing pipeline from S3
        success = pipeline.indexing_pipeline(
            use_s3=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
        )

        if success:
            logger.info("=" * 80)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            sys.exit(0)
        else:
            logger.error("✗ PIPELINE FAILED")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
