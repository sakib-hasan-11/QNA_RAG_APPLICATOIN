from pinecone import Pinecone
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log import get_logger

logger = get_logger(__name__)

def create_DB():
    """
    Create and connect to Pinecone database.
    
    Returns:
        tuple: (Pinecone client, index object)
    """
    try:
        pc = Pinecone(
            api_key=os.environ['PINECONE_API_KEY']
        )

        index = pc.Index(
            host=os.environ["PINECONE_HOST"]
        )
        
        logger.info('Successfully connected to Pinecone database')
        logger.info(f'Index stats: {index.describe_index_stats()}')
        
        return pc, index
        
    except KeyError as e:
        logger.error(f'Missing environment variable: {e}')
        raise
    except Exception as e:
        logger.error(f'Error connecting to Pinecone: {e}')
        raise

