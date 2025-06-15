# config.py
from datetime import timedelta

CONFIG = {
    "models": [
        # "qwen3:8b",
        # "qwen3:4b",
        # "gemma3:4b-it-qat",
        # "gemma3:4b",
        # "gemma3:1b",
        "gemma3-1b-it-qat:latest",
        # "gemma3-1b-ftr-10k:latest",
        # "gemma3-1b-ft-25k:latest",
        # "gemma3-1b-ft-75k:latest",
        "gemma3-1b-ftr-75k:latest",
        # "yandex/YandexGPT-5-Lite-8B-instruct-GGUF"
        # "qwen3:1.7b"
    ],
    "max_runtime": timedelta(minutes=10),  # Время тестирования в минутах
    "dataset": {
        "name": "RussianNLP/Mixed-Summarization-Dataset",
        "split": "train"
    },
    "output": {
        "metrics_file": "metrics.json",
        "backup_metrics_file": "backup_metrics.json",
        "failed_summaries_file": "failed_summaries.json",
        "results_dir": "results"
    },
    "save_interval": 20,  # Сохранять каждые N текстов
    "ollama_api_url": "http://localhost:7869/api/generate"
}