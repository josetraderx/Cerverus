from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Cerverus API"
    debug: bool = True

settings = Settings()
