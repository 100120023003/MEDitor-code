import os


config = {
    "api_type": os.getenv("OPENAI_API_TYPE", "openai"),
    "api_base": os.getenv("OPENAI_API_BASE", ""),
    "api_version": os.getenv("OPENAI_API_VERSION", ""),
    "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
}
