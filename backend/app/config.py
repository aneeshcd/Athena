from functools import lru_cache
import os
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
DOCKER_DEFAULT_NEO4J_URI = "bolt://neo4j:7687"


class Settings(BaseSettings):
    app_name: str = "Athena SE"
    llm_provider: str = "ollama"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_extraction_model: str = "gpt-4.1-mini"
    openai_summary_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_ms: int = 300000
    ollama_max_context_nodes: int = 20
    ollama_max_context_edges: int = 30
    ollama_max_description_chars: int = 180
    ollama_warmup_on_start: bool = False
    neo4j_uri: str = Field(DOCKER_DEFAULT_NEO4J_URI, validation_alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_USER"))
    neo4j_password: str = Field("athena-password", validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", validation_alias="NEO4J_DATABASE")
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ENV_FILE, BACKEND_ENV_FILE, ".env"),
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def neo4j_username(self) -> str:
        return self.neo4j_user


@lru_cache
def get_settings() -> Settings:
    return Settings()


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER") == "1"
