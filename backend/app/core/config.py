# backend/app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Driver Drowsiness Detection API"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000"]
    database_url: str = "sqlite:///./dev.db"
    secret_key: str = "change-this-later"
    access_token_expire_minutes: int = 60
    model_path: str = "../storage/models/drowsiness_bilstm.pt"

settings = Settings()