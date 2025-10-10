import os
from dotenv import load_dotenv
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "configs" / "app"

load_dotenv(ROOT / ".env")
env = os.getenv("APP_ENV")


class AppConfig(BaseModel):
    name: str
    version: str


class ChatConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    retry: int


class EmbeddingConfig(BaseModel):
    model: str


class LoggingConfig(BaseModel):
    level: str = "INFO"


class AppSettings(BaseSettings):
    APP_ENV: str | None = "dev"
    OPENAI_API_KEY: str | None = None

    # YAML 설정
    app: AppConfig | None = None
    llm: ChatConfig | None = None
    embedding: EmbeddingConfig | None = None
    logging: LoggingConfig | None = None

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_settings = YamlConfigSettingsSource(
            settings_cls, yaml_file=YAML_PATH / f"{env}.yaml"
        )
        base_yaml_settings = YamlConfigSettingsSource(
            settings_cls, yaml_file=YAML_PATH / "base.yaml"
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            base_yaml_settings,
            file_secret_settings
        )


settings = AppSettings()

if __name__ == "__main__":
    print(settings.model_dump())
