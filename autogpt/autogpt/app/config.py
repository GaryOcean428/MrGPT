from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, List

import forge
# NOTE: The following imports are unresolved. Ensure the 'forge' package
# is properly installed and accessible in the project environment.
from forge.config.base import BaseConfig
from forge.logging.config import LoggingConfig
from forge.models.config import Configurable
from pydantic import ValidationError, validator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(forge.__file__).parent.parent
AZURE_CONFIG_FILE = Path("azure.yaml")


# Define the custom ModelName Enum with your models
class ModelName(str, Enum):
    CLAUDE_35_SONNET_20240620 = "claude-3-5-sonnet-20240620"
    LLAMA3_70B_VERSATILE = "llama-3.1-70b-versatile"
    LLAMA3_8B_INSTANT = "llama-3.1-8b-instant"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    # Add other models as needed


def _safe_split(s: Optional[str], sep: str = ",") -> List[str]:
    """
    Split a string by a separator. Return an empty list if the string is None.
    """
    if s is None:
        return []
    return s.split(sep)


class AppConfig(BaseConfig):
    def __init__(self, **data):
        super().__init__(**data)
        self.name = data.get("name", "Auto-GPT configuration")
        self.description = data.get(
            "description",
            "Default configuration for the Auto-GPT application."
        )
        self.project_root = data.get("project_root", PROJECT_ROOT)
        self.app_data_dir = data.get("app_data_dir", PROJECT_ROOT / "data")
        self.skip_news = data.get("skip_news", False)
        self.skip_reprompt = data.get("skip_reprompt", False)
        self.authorise_key = data.get(
            "authorise_key",
            os.getenv("AUTHORISE_COMMAND_KEY", "y")
        )
        self.exit_key = data.get("exit_key", os.getenv("EXIT_KEY", "n"))
        self.noninteractive_mode = data.get("noninteractive_mode", False)
        self.logging = data.get("logging", LoggingConfig())
        self.component_config_file = data.get("component_config_file")
        self.fast_llm = data.get("fast_llm", ModelName.LLAMA3_8B_INSTANT)
        self.smart_llm = data.get(
            "smart_llm",
            ModelName.CLAUDE_35_SONNET_20240620
        )
        self.temperature = data.get("temperature", 0.5)
        self.openai_functions = data.get("openai_functions", True)
        self.embedding_model = data.get(
            "embedding_model",
            "text-embedding-3-small"
        )
        self.continuous_mode = data.get("continuous_mode", False)
        self.continuous_limit = data.get("continuous_limit", 0)
        self.disabled_commands = data.get(
            "disabled_commands",
            _safe_split(os.getenv("DISABLED_COMMANDS"))
        )
        self.restrict_to_workspace = data.get("restrict_to_workspace", True)
        self.openai_credentials = data.get("openai_credentials")
        self.azure_config_file = data.get(
            "azure_config_file",
            AZURE_CONFIG_FILE
        )

    @validator("openai_functions")
    def validate_openai_functions(cls, value, values):
        if value:
            smart_llm = values.get("smart_llm")
            if smart_llm is None:
                return value
            if smart_llm not in [ModelName.GPT4O, ModelName.GPT4O_MINI]:
                message = (
                    f"Model {smart_llm} does not support tool calling. "
                    "Disable OPENAI_FUNCTIONS or choose a suitable model."
                )
                raise ValueError(message)
        return value


class ConfigBuilder(Configurable[AppConfig]):
    default_settings = AppConfig()

    @classmethod
    def build_config_from_env(
        cls,
        project_root: Path = PROJECT_ROOT
    ) -> AppConfig:
        """Initialize the Config class"""
        config = cls.build_agent_configuration()
        config.project_root = project_root
        config.azure_config_file = project_root / config.azure_config_file
        return config


async def assert_config_has_required_llm_api_keys(config: AppConfig) -> None:
    """
    Check if API keys are set for the configured SMART_LLM and FAST_LLM.
    """
    models = {config.smart_llm, config.fast_llm}

    # For LLama models via Groq
    if any(
        model in models
        for model in [
            ModelName.LLAMA3_70B_VERSATILE,
            ModelName.LLAMA3_8B_INSTANT
        ]
    ):
        try:
            from forge.llm.providers.groq import GroqProvider
            groq = GroqProvider()
            await groq.get_available_models()
        except Exception as e:
            logger.error("Groq models are unavailable or misconfigured.")
            raise e

    # For GPT-4o models via OpenAI
    if any(model in models for model in [ModelName.GPT4O, ModelName.GPT4O_MINI]):
        try:
            from forge.llm.providers.openai import OpenAIProvider
            openai = OpenAIProvider()
            await openai.get_available_models()
        except ValidationError as e:
            if "api_key" not in str(e):
                raise
            logger.error(
                "Set your OpenAI API key in .env or as an environment "
                "variable"
            )
            raise ValueError("OpenAI is unavailable: can't load credentials")
        except Exception as e:
            logger.error("OpenAI models are unavailable or misconfigured.")
            raise e

    # For Anthropic models
    if ModelName.CLAUDE_35_SONNET_20240620 in models:
        try:
            from forge.llm.providers.anthropic import AnthropicCredentials
            AnthropicCredentials.from_env()
        except ValidationError as e:
            if "api_key" in str(e):
                logger.error(
                    "Set your Anthropic API key in .env or as an "
                    "environment variable"
                )
                raise ValueError(
                    "Anthropic is unavailable: can't load credentials"
                ) from e
            else:
                raise
        except Exception as e:
            logger.error("Anthropic models are unavailable or misconfigured.")
            raise e
