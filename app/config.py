from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_KEY: str
    APP_ENV: str = "development"
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
