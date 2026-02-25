import os
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from enum import Enum


class ModelType(Enum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    GENERAL = "general"


MODEL_CLASS_MAP = {
    ModelType.GENERAL: AzureOpenAI,
    ModelType.COMPLETION: AzureChatOpenAI,
    ModelType.EMBEDDING: AzureOpenAIEmbeddings,
}


def create_openai_model_instance(
    azure_openai_deployment: str = None,
    azure_openai_endpoint: str = None,
    azure_openai_key: str = None,
    azure_openai_api_version: str = None,
    model_type: ModelType = ModelType.COMPLETION,
    **kwargs,
) -> AzureChatOpenAI | AzureOpenAIEmbeddings | AzureOpenAI:
    """
    Factory function to create an instance of AzureChatOpenAI or AzureOpenAIEmbeddings
    based on environment variables.
    """

    # Check if using identity-based authentication
    token_provider = None
    if not azure_openai_key:
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )

    model_args = {
        "azure_deployment": azure_openai_deployment,
        "azure_endpoint": azure_openai_endpoint,
        "api_key": azure_openai_key,  # Optional if using identity-based auth
        "azure_ad_token_provider": token_provider,
        "api_version": azure_openai_api_version or "2024-08-01-preview",
    }

    # Create and return the appropriate model instance based on the model type
    try:
        return MODEL_CLASS_MAP[model_type](**model_args, **kwargs)
    except KeyError:
        raise ValueError(f"Unsupported model type: {model_type}")
