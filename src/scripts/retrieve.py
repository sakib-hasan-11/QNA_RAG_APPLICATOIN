import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)

def retrieve_query(query, embedding_model, index, k=5): 
    """
    Retrieve top-k most similar documents from Pinecone based on cosine similarity.
    
    Args:
        query (str): The input query text
        embedding_model: The embedding model to use for query embedding
        index: The Pinecone index to query
        k (int): Number of top similar results to return (default: 5)
    
    Returns:
        list: List of matching results with scores, text, and metadata
    """
    try:
        # Embed the query
        query_embedding = embedding_model.embed_query(query)
        logger.info('Query embedded successfully')
        
        # Query Pinecone index
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Format the results
        matching_results = []
        for match in results['matches']:
            result = {
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'source': match['metadata'].get('source', ''),
                'page': match['metadata'].get('page', 0)
            }
            matching_results.append(result)
        
        logger.info(f'Retrieved {len(matching_results)} matching documents')
        return matching_results
        
    except Exception as e:
        logger.error(f'Error retrieving documents: {e}')
        raise