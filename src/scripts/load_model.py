import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from utils.log import get_logger

logger = get_logger(__name__)


def load_embed_model(model_name="text-embedding-3-small"):
    model = None
    try:
        model = OpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"
        )
    except KeyError as e:
        logger.info(f"embed model not loaeded..\n {e}")
        raise
    except Exception as e:
        logger.info(f"embed model not loaeded..\n {e}")
        return None

    return model


def load_llm(model_name="gpt-4o-mini"):
    model = None
    try:
        model = ChatOpenAI(
            model=model_name,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=1,
            max_completion_tokens=500,
        )
    except KeyError as e:
        logger.info(f"llm not loaded... {e}")
        raise
    except Exception as e:
        logger.info(f"llm not loaded... {e}")
        return None

    return model
