from langchain_openai import OpenAIEmbeddings,ChatOpenAI
import os 
from utils.log import get_logger

logger = get_logger(__name__)



def load_embed_model(model_name='text-embedding-3-small'):
    try:
        model = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model = 'text-embedding-3-small'
        )
    except Exception as e:
        logger.info(f'embed model not loaeded..\n {e}')

    return model







def load_llm(model_name='gpt-4o-mini'):
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=os.environ['OPENAI_API_KEY'],
            temperature=1,
            max_completion_tokens=500
        )
    except Exception as e : 
        logger.info(f'llm not loaded... {e}')

    return model

