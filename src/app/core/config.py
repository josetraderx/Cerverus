from pydantic import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Cerverus Fraud Detection API"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()