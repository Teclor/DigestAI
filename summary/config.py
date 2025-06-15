import os
from pydantic import BaseModel
import json

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "resources/config.json")

class KafkaConfig(BaseModel):
    bootstrap_servers: str = "kafka:9092"
    max_messages_to_read: int = 1000
    poll_timeout_ms: int = 1000
    max_poll_records: int = 500
    fetch_max_bytes: int = 52428800
    max_partition_fetch_bytes: int = 10485760


class OllamaConfig(BaseModel):
    url: str = "http://ollama:11434/api/generate"
    default_model: str = "gemma3-1b-ftr-75k:latest"
    model: str = "gemma3-1b-ftr-75k-lowtemp:latest"
    models: list = [
        "gemma3-1b-it-qat:latest",
        "gemma3-1b-ftr-10k:latest",
        "gemma3-1b-ftr-75k:latest",
        "gemma3-1b-ftr-75k-lowtemp"
    ]
    default_tokenizer: str = "unsloth/gemma-3-1b-it-qat"
    timeout: int = 20
    max_input_text_tokens: int = 400


class AppConfig(BaseModel):
    kafka: KafkaConfig = KafkaConfig()
    ollama: OllamaConfig = OllamaConfig()


def load_config() -> AppConfig:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)

    return AppConfig(**config_data)

config = load_config()