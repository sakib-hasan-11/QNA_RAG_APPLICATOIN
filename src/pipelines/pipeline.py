"""
RAG Pipeline - Complete implementation for document indexing and querying.

This module provides two main pipelines:
1. Indexing Pipeline: Load documents, chunk them, embed, and store in Pinecone
2. Query Pipeline: Retrieve relevant documents and generate answers using LLM
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.chunks import divide_chunks
from scripts.doc_reader import read_doc
from scripts.docker_ecr import DockerECRManager
from scripts.load_model import load_embed_model, load_llm
from scripts.rag_query import rag_query
from scripts.vector_sender import sent_vector
from utils.log import get_logger
from utils.pinecone_DB import create_DB

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline for document indexing and querying.
    """

    def __init__(self):
        """Initialize RAG pipeline with models and database connection."""
        self.embedding_model = None
        self.llm_model = None
        self.pc = None
        self.index = None
        logger.info("RAG Pipeline initialized")

    def setup_models(
        self, embed_model_name="text-embedding-3-small", llm_model_name="gpt-4o-mini"
    ):
        """
        Load embedding and LLM models.

        Args:
            embed_model_name (str): Name of the embedding model
            llm_model_name (str): Name of the LLM model
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 1: Loading Models")
            logger.info("=" * 80)

            # Load embedding model
            logger.info(f"Loading embedding model: {embed_model_name}")
            self.embedding_model = load_embed_model(embed_model_name)
            logger.info("✓ Embedding model loaded successfully")

            # Load LLM model
            logger.info(f"Loading LLM model: {llm_model_name}")
            self.llm_model = load_llm(llm_model_name)
            logger.info("✓ LLM model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def setup_database(self):
        """
        Connect to Pinecone database.
        """
        try:
            logger.info("=" * 80)
            logger.info("STEP 2: Connecting to Pinecone Database")
            logger.info("=" * 80)

            self.pc, self.index = create_DB()
            logger.info("✓ Successfully connected to Pinecone database")

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def deploy_to_ecr(
        self, ecr_repo_name="rag-images-container", aws_region=None, root_dir=None
    ):
        """
        Build Docker images and push them to ECR.

        Args:
            ecr_repo_name (str): ECR repository name (default: rag-images-container)
            aws_region (str): AWS region (defaults to AWS_REGION env var)
            root_dir (str): Root directory of the project (auto-detected if None)

        Returns:
            dict: Contains pushed image URIs and metadata
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STEP 6: Building and Pushing Docker Images to ECR")
            logger.info("=" * 80)

            # Initialize Docker ECR Manager
            docker_manager = DockerECRManager(
                aws_region=aws_region, ecr_repo_name=ecr_repo_name
            )

            # Build and push images
            result = docker_manager.build_and_push_images(root_dir=root_dir)

            logger.info("✓ Docker images successfully deployed to ECR")
            return result

        except Exception as e:
            logger.error(f"Error deploying to ECR: {e}")
            raise

    def indexing_pipeline(
        self,
        data_dir=None,
        chunk_size=800,
        chunk_overlap=80,
        batch_size=100,
        use_s3=False,
        deploy_images=True,
        ecr_repo_name="rag-images-container",
        aws_region=None,
    ):
        """
        Complete indexing pipeline: Load docs → Chunk → Embed → Store in Pinecone → Deploy to ECR.

        Args:
            data_dir (str): Directory containing PDF files (for local files)
            chunk_size (int): Size of each chunk (default: 800)
            chunk_overlap (int): Overlap between chunks (default: 80)
            batch_size (int): Batch size for vector upserting (default: 100)
            use_s3 (bool): If True, load documents from S3 instead of local directory
            deploy_images (bool): If True, build and push Docker images to ECR (default: True)
            ecr_repo_name (str): ECR repository name (default: rag-images-container)
            aws_region (str): AWS region for ECR (defaults to AWS_REGION env var)

        Returns:
            dict: Contains success status and ECR deployment info (if deployed)
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING INDEXING PIPELINE")
            logger.info("=" * 80 + "\n")

            # Step 1: Setup models
            if self.embedding_model is None or self.llm_model is None:
                self.setup_models()

            # Step 2: Setup database
            if self.index is None:
                self.setup_database()

            # Step 3: Load documents
            logger.info("=" * 80)
            logger.info("STEP 3: Loading Documents")
            logger.info("=" * 80)
            if use_s3:
                logger.info("Loading documents from S3 bucket")
                docs = read_doc(use_s3=True)
            else:
                logger.info(f"Reading documents from: {data_dir}")
                docs = read_doc(data_dir, use_s3=False)
            logger.info(f"✓ Loaded {len(docs)} documents")

            # Step 4: Chunk documents
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Chunking Documents")
            logger.info("=" * 80)
            logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
            chunked_docs = divide_chunks(
                docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            logger.info(f"✓ Created {len(chunked_docs)} chunks")

            # Step 5: Embed and store in Pinecone
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: Embedding and Storing Vectors")
            logger.info("=" * 80)
            logger.info(f"Batch size: {batch_size}")
            sent_vector(chunked_docs, self.embedding_model, batch_size, self.index)

            # Step 6: Deploy Docker images to ECR (optional)
            ecr_result = None
            if deploy_images:
                try:
                    ecr_result = self.deploy_to_ecr(
                        ecr_repo_name=ecr_repo_name, aws_region=aws_region
                    )
                except Exception as e:
                    logger.warning(f"ECR deployment failed (continuing anyway): {e}")
                    logger.warning(
                        "Pipeline completed successfully, but images were not deployed to ECR"
                    )

            logger.info("\n" + "=" * 80)
            logger.info("✓ INDEXING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80 + "\n")

            return {"success": True, "ecr_deployment": ecr_result}

        except Exception as e:
            logger.error(f"Error in indexing pipeline: {e}")
            logger.error("✗ INDEXING PIPELINE FAILED")
            raise

    def query_pipeline(self, query, k=3, verbose=True):
        """
        Complete query pipeline: Retrieve → Create prompt → Generate answer.

        Args:
            query (str): User's question
            k (int): Number of relevant documents to retrieve (default: 3)
            verbose (bool): Print intermediate steps (default: True)

        Returns:
            dict: Contains 'answer', 'sources', and 'retrieved_docs'
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STARTING QUERY PIPELINE")
            logger.info("=" * 80 + "\n")

            # Ensure models and database are initialized
            if self.embedding_model is None or self.llm_model is None:
                logger.info("Models not loaded, loading now...")
                self.setup_models()

            if self.index is None:
                logger.info("Database not connected, connecting now...")
                self.setup_database()

            # Execute RAG query
            result = rag_query(
                query=query,
                embedding_model=self.embedding_model,
                index=self.index,
                LLM_model=self.llm_model,
                k=k,
                verbose=verbose,
            )

            logger.info("\n" + "=" * 80)
            logger.info("✓ QUERY PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80 + "\n")

            return result

        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            logger.error("✗ QUERY PIPELINE FAILED")
            raise


# Convenience functions for direct use
def run_indexing(
    data_dir,
    chunk_size=800,
    chunk_overlap=80,
    batch_size=100,
    deploy_images=True,
    ecr_repo_name="rag-images-container",
):
    """
    Run the indexing pipeline.

    Args:
        data_dir (str): Directory containing PDF files
        chunk_size (int): Size of each chunk (default: 800)
        chunk_overlap (int): Overlap between chunks (default: 80)
        batch_size (int): Batch size for vector upserting (default: 100)
        deploy_images (bool): If True, build and push Docker images to ECR (default: True)
        ecr_repo_name (str): ECR repository name (default: rag-images-container)

    Returns:
        dict: Contains success status and ECR deployment info
    """
    pipeline = RAGPipeline()
    return pipeline.indexing_pipeline(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        batch_size=batch_size,
        deploy_images=deploy_images,
        ecr_repo_name=ecr_repo_name,
    )


def run_query(query, k=3, verbose=True):
    """
    Run a RAG query.

    Args:
        query (str): User's question
        k (int): Number of relevant documents to retrieve (default: 3)
        verbose (bool): Print intermediate steps (default: True)

    Returns:
        dict: Contains 'answer', 'sources', and 'retrieved_docs'
    """
    pipeline = RAGPipeline()
    return pipeline.query_pipeline(query=query, k=k, verbose=verbose)


if __name__ == "__main__":
    """
    Example usage of the RAG pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Index documents or query"
    )
    parser.add_argument(
        "mode", choices=["index", "query"], help="Pipeline mode: index or query"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Directory containing PDF files (for indexing)"
    )
    parser.add_argument("--query", type=str, help="Query text (for querying)")
    parser.add_argument(
        "--chunk_size", type=int, default=800, help="Chunk size (default: 800)"
    )
    parser.add_argument(
        "--chunk_overlap", type=int, default=80, help="Chunk overlap (default: 80)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for upserting (default: 100)",
    )
    parser.add_argument(
        "--k", type=int, default=3, help="Number of documents to retrieve (default: 3)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    if args.mode == "index":
        if not args.data_dir:
            logger.error("--data_dir is required for indexing mode")
            exit(1)

        run_indexing(
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
        )

    elif args.mode == "query":
        if not args.query:
            logger.error("--query is required for query mode")
            exit(1)

        result = run_query(query=args.query, k=args.k, verbose=args.verbose)
        print("\n" + "=" * 80)
        print("QUERY RESULT")
        print("=" * 80)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {result['sources']}")
