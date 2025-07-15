import time
from functools import cached_property
from typing import Dict, List

import openai
from joblib import Parallel, delayed
from pydantic import BaseModel, ConfigDict, PositiveInt

# Local imports
from utils import logger as utils_logger

# Initialize logger
logger = utils_logger.logger_config("llm")


class LLMClient(BaseModel):
    # Configuration to forbid extra parameters
    model_config = ConfigDict(extra="forbid")
    nthreads: PositiveInt

    @classmethod
    def create(cls, client_type: str, **kwargs) -> "LLMClient":
        """Factory method to create instances of different client types."""
        if client_type in CLIENTS:
            return CLIENTS[client_type](**kwargs)
        raise ValueError(f"Unsupported client type: {client_type}")

    def chat(self, *args, **kwargs) -> str:
        """Method to be implemented by subclasses for sending chat requests."""
        raise NotImplementedError

    def chat_many(
        self, messages_list: List[List[Dict[str, str]]], temperature: float
    ) -> List[str]:
        """
        Send multiple chat requests in parallel using threading.

        Parameters
        ----------
        messages_list: List[List[Dict[str, str]]]
            List of messages for each chat request.

        temperature: float
            Temperature setting for the model.

        Returns
        -------
        List[str]
            List of responses from the chat requests.
        """
        pool = Parallel(n_jobs=self.nthreads, backend="threading")
        return pool(
            delayed(self.chat)(i, m, temperature) for i, m in enumerate(messages_list)
        )


class OpenAIClient(LLMClient):
    model_name: str
    api_key: str

    @cached_property
    def client(self) -> openai.OpenAI:
        """Initialize and cache the OpenAI client."""
        return openai.OpenAI(api_key=self.api_key)

    def chat(self, i: int, prompt: List[Dict[str, str]], temperature: float = 0) -> str:
        """Send a chat request to the OpenAI API with retry logic."""
        max_retries = 5
        retry_delay = 1
        logger.info(f"Sending request {i} to OpenAI API")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=prompt, temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for request {i}: {e}"
                )
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error("Max retries reached for request. Giving up.")
                    return ""


class AzureOpenAIClient(LLMClient):
    model_name: str
    api_key: str
    url: str
    api_version: str

    @cached_property
    def client(self) -> openai.AzureOpenAI:
        """Initialize and cache the Azure OpenAI client."""
        return openai.AzureOpenAI(
            api_key=self.api_key, azure_endpoint=self.url, api_version=self.api_version
        )

    def chat(self, i: int, prompt: List[Dict[str, str]], temperature: float) -> str:
        """Send a chat request to the Azure OpenAI API with retry logic."""
        max_retries = 5
        retry_delay = 1
        logger.info(f"Sending request {i} to Azure OpenAI")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=prompt, temperature=temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for request {i}: {e}"
                )
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2**attempt)
                    logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for request {i}. Giving up.")
                    return ""


# Dictionary mapping client identifiers to client classes
CLIENTS = {
    "openai": OpenAIClient,
    "azureopenai": AzureOpenAIClient,
}
