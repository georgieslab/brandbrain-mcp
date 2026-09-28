"""AWS Bedrock client integration using boto3.

Supports:
- Moonshot AI Kimi K3 Converse API (model ID: moonshotai.kimi-k3)
- Amazon Titan Text Embeddings V2 with local fallback
- Robust credential handling from environment variables
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

logger = logging.getLogger("brandbrain.bedrock")

DEFAULT_CHAT_MODEL = os.getenv(
    "BEDROCK_MODEL_ID", "moonshotai.kimi-k3"
)
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
DEFAULT_REGION = (
    os.getenv("AWS_DEFAULT_REGION")
    or os.getenv("AWS_REGION")
    or "us-east-1"
)


class BedrockClient:
    """Wrapper around AWS Bedrock Runtime for Converse API and Embeddings."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        embedding_model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.model_id = model_id or DEFAULT_CHAT_MODEL
        self.embedding_model_id = embedding_model_id or DEFAULT_EMBEDDING_MODEL
        self.region_name = region_name or DEFAULT_REGION

        if client is not None:
            self._client = client
        else:
            self._client = self._initialize_client()

    def _initialize_client(self) -> Optional[Any]:
        """Initialize the boto3 Bedrock Runtime client."""
        try:
            config = Config(
                region_name=self.region_name,
                retries={"max_attempts": 3, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
            )
            # boto3 will naturally resolve env vars:
            # AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, etc.
            client = boto3.client(
                service_name="bedrock-runtime",
                region_name=self.region_name,
                config=config,
            )
            return client
        except (BotoCoreError, Exception) as exc:
            logger.warning(
                "Failed to initialize AWS Bedrock client: %s. Local fallback will be used if needed.",
                exc,
            )
            return None

    @property
    def client(self) -> Any:
        """Returns the boto3 client, attempting initialization if not yet created."""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client

    def has_credentials(self) -> bool:
        """Checks if AWS credentials appear to be configured in environment or boto3 session."""
        try:
            session = boto3.Session()
            creds = session.get_credentials()
            return creds is not None and bool(creds.access_key)
        except Exception:
            return False

    def converse(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_id: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Send a message to the LLM (moonshotai.kimi-k3) using Bedrock Converse API.

        Args:
            prompt: User message prompt.
            system_prompt: Optional system instructions.
            model_id: Optional model override.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens in response.

        Returns:
            The text response from the model.
        """
        active_model = model_id or self.model_id
        client = self.client

        if client is None:
            raise RuntimeError(
                "AWS Bedrock client is not initialized. Please configure AWS credentials."
            )

        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]

        converse_kwargs: Dict[str, Any] = {
            "modelId": active_model,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }

        if system_prompt:
            converse_kwargs["system"] = [{"text": system_prompt}]

        try:
            response = client.converse(**converse_kwargs)
            output = response.get("output", {})
            message = output.get("message", {})
            content_list = message.get("content", [])
            for part in content_list:
                if "text" in part:
                    return part["text"]
            return ""
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.error("AWS Bedrock Converse API call failed for model %s: %s", active_model, exc)
            raise RuntimeError(f"Bedrock Converse error: {exc}") from exc

    def get_embedding(
        self,
        text: str,
        dimensions: int = 1024,
        normalize: bool = True,
    ) -> List[float]:
        """Generate text embedding using Amazon Titan Text Embeddings V2.

        Args:
            text: Input text to embed.
            dimensions: Output embedding dimension (e.g., 256, 512, 1024).
            normalize: Whether to normalize embedding vector.

        Returns:
            List of floats representing the embedding vector.
        """
        client = self.client
        if client is None:
            raise RuntimeError("Bedrock client unavailable for embeddings.")

        payload = {
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize,
        }

        try:
            response = client.invoke_model(
                modelId=self.embedding_model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload),
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body["embedding"]
        except (ClientError, BotoCoreError, NoCredentialsError, Exception) as exc:
            logger.debug(
                "Titan Embeddings call failed (%s). Raising to trigger vector store fallback.",
                exc,
            )
            raise RuntimeError(f"Titan embedding error: {exc}") from exc


def extract_json_from_response(text: str) -> Dict[str, Any]:
    """Helper to extract and parse JSON payload from LLM responses."""
    text_clean = text.strip()

    # Try direct parse first
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass

    # Try finding markdown code block ```json ... ``` or ``` ... ```
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try matching first { to last }
    first_brace = text_clean.find("{")
    last_brace = text_clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = text_clean[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from text: {text[:200]}...")

