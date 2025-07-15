import time
from functools import cached_property
from typing import List, Union

import openai
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, PositiveInt
from sentence_transformers import SentenceTransformer

from utils import logger as utils_logger

# Initialize logger for embedding operations
logger = utils_logger.logger_config("embedding")


class EmbeddingClient(BaseModel):
    """Base class for embedding clients."""

    model_config = ConfigDict(extra="forbid")
    nthreads: PositiveInt

    @classmethod
    def create(cls, client_type: str, **kwargs) -> "EmbeddingClient":
        """Factory method to create client instances."""
        if client_type in CLIENTS:
            return CLIENTS[client_type](**kwargs)
        raise ValueError(f"Unsupported client type: {client_type}")

    def embed(self, i: int, text: Union[str, List[str]]) -> List[float]:
        """Embed a single text or a list of texts."""
        raise NotImplementedError("This method must be overridden by subclasses.")

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using parallel processing."""
        pool = Parallel(n_jobs=self.nthreads, backend="threading")
        return pool(delayed(self.embed)(i, text) for i, text in enumerate(texts))


class OpenAIClient(EmbeddingClient):
    """Client for embedding via OpenAI API."""

    api_key: str
    model_name: str

    @cached_property
    def client(self) -> openai.OpenAI:
        """Create OpenAI client instance."""
        return openai.OpenAI(api_key=self.api_key)

    def embed(self, i: int, text: Union[str, List[str]]) -> List[float]:
        """Embed text using OpenAI API with retry logic."""
        if isinstance(text, list) and not text:
            return []
        max_retries = 5
        retry_delay = 1  # Initial delay in seconds
        logger.info(f"Embedding index: {i}")

        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text, model=self.model_name
                )
                embeddings = (
                    [r.embedding for r in response.data]
                    if isinstance(text, list)
                    else response.data[0].embedding
                )
                return embeddings
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached. Giving up.")
                    return []


class LocalEmbeddingClient(EmbeddingClient):
    """Client for embedding via local SentenceTransformer model."""

    model_name: str

    @cached_property
    def model(self) -> SentenceTransformer:
        """Load SentenceTransformer model."""
        return SentenceTransformer(self.model_name)

    def embed(self, i: int, text: Union[str, List[str]]) -> List[float]:
        """Embed text using local model with retry logic."""
        if isinstance(text, list) and not text:
            return []
        max_retries = 5
        retry_delay = 1
        logger.info(f"Embedding index: {i}")

        for attempt in range(max_retries):
            try:
                embeddings = (
                    self.model.encode(
                        text, normalize_embeddings=True, show_progress_bar=False
                    ).tolist()
                    if isinstance(text, list)
                    else self.model.encode(
                        [text], normalize_embeddings=True, show_progress_bar=False
                    )[0].tolist()
                )
                return embeddings
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached. Giving up.")
                    return []


# Mapping for client types to their respective classes
CLIENTS = {"openai": OpenAIClient, "local": LocalEmbeddingClient}
