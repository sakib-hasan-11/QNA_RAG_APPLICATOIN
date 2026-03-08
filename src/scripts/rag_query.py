import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompt import create_prompt
from scripts.retrieve import retrieve_query
from utils.log import get_logger

logger = get_logger(__name__)


def rag_query(query, embedding_model, index, LLM_model, k=3, verbose=True):
    """
    Main RAG function: Retrieve relevant documents and generate an answer.

    Args:
        query (str): User's question
        embedding_model: The embedding model for query encoding
        index: The Pinecone index to search
        LLM_model: The language model for generating answers
        k (int): Number of documents to retrieve (default: 3)
        verbose (bool): Print intermediate steps (default: True)

    Returns:
        dict: Contains 'answer', 'sources', and 'retrieved_docs'
    """
    try:
        # Step 1: Retrieve relevant documents
        if verbose:
            logger.info("🔍 Searching for relevant documents...")

        retrieved_docs = retrieve_query(query, embedding_model, index, k=k)

        if verbose:
            logger.info(f"✓ Retrieved {len(retrieved_docs)} documents\n")

        # Step 2: Create prompt with context
        prompt = create_prompt(query, retrieved_docs)

        if verbose:
            logger.info("✓ Generated prompt with context\n")

        # Step 3: Get LLM response
        if verbose:
            logger.info("✓ Generating answer...\n")

        answer = LLM_model.invoke(prompt)

        # Step 4: Format sources
        sources = [
            {"source": doc["source"], "page": doc["page"], "score": doc["score"]}
            for doc in retrieved_docs
        ]

        # Display results
        if verbose:
            logger.info("=" * 80)
            logger.info(f"Question: {query}")
            logger.info("=" * 80)
            logger.info(f"\n📌 Answer:\n{answer}\n")
            logger.info("-" * 80)
            logger.info("\n📚 Sources:")
            for i, source in enumerate(sources, 1):
                logger.info(
                    f"  {i}. {source['source']} (Page {source['page']}, Score: {source['score']:.4f})"
                )
            logger.info("=" * 80)

        return {"answer": answer, "sources": sources, "retrieved_docs": retrieved_docs}

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise
