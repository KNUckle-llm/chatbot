from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    APP_NAME: str = "KNUC API Server"
    APP_VERSION: str = "0.1.0"
    OPENAI_API_KEY: str | None = None
    VECTORSTORE_DIR: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = AppSettings()

if __name__ == "__main__":
    print(settings.model_dump())
