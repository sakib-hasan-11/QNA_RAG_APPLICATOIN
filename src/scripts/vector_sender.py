import os
import sys

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)


def sent_vector(chunk_docs, embedding_model, batch_size, index):
    """
    Embed document chunks and send vectors to Pinecone.
    
    Args:
        chunk_docs (list): List of chunked documents
        embedding_model: The embedding model to use
        batch_size (int): Number of vectors to batch before upserting
        index: The Pinecone index to upsert to
    """
    try:
        vectors_to_upsert = []

        logger.info(f"Processing {len(chunk_docs)} document chunks...")

        for i, doc in enumerate(tqdm(chunk_docs, desc="Embedding chunks")):
            # Generate embedding for the document
            embedding = embedding_model.embed_query(doc.page_content)

            # Prepare metadata
            metadata = {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
            }

            # Create vector tuple (id, embedding, metadata) for each vector.
            vector_id = f"doc_{i}"
            vectors_to_upsert.append((vector_id, embedding, metadata))

            # Upsert in batches
            if len(vectors_to_upsert) >= batch_size:
                index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Upserted batch of {len(vectors_to_upsert)} vectors")
                vectors_to_upsert = []

        # If last batch is less than batch_size, upsert remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted final batch of {len(vectors_to_upsert)} vectors")

        logger.info("✓ All documents embedded and stored in Pinecone!")
        logger.info(f"Index stats: {index.describe_index_stats()}")

    except Exception as e:
        logger.error(f"Error sending vectors to Pinecone: {e}")
        raise
