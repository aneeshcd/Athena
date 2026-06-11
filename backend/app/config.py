from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Athena SE"
    openai_api_key: str | None = None
    openai_extraction_model: str = "gpt-4.1-mini"
    openai_summary_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = Field("neo4j", validation_alias=AliasChoices("NEO4J_USERNAME", "NEO4J_USER"))
    neo4j_password: str = "athena-password"
    neo4j_database: str = "neo4j"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def neo4j_username(self) -> str:
        return self.neo4j_user


@lru_cache
def get_settings() -> Settings:
    return Settings()
