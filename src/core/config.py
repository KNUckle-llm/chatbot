import os
from dotenv import load_dotenv
from pathlib import Path

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


class AppSettings(BaseSettings):

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="allow")

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
            settings_cls,
            yaml_file=[YAML_PATH / "base.yaml", YAML_PATH / f"{env}.yaml",]
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_settings,
            file_secret_settings
        )


settings = AppSettings().model_dump()

if __name__ == "__main__":
    print(settings)
