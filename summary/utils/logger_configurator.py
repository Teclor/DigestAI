import os
import json
from loguru import logger


class LoggerConfigurator:
    LOG_DIR = "logs"
    FORMAT_SIMPLE = "{time} | {level} | {module}:{line} | {message}"
    FORMAT_JSON = "{message}"

    def __init__(self, log_file_name: str, log_format: str = "simple"):
        self.log_file_name = log_file_name
        self.log_format = log_format.lower()
        self.full_log_path = os.path.join(LoggerConfigurator.LOG_DIR, log_file_name)

        os.makedirs(LoggerConfigurator.LOG_DIR, exist_ok=True)
        logger.remove()

        if self.log_format == "json":
            self._setup_json_logging()
        elif self.log_format == "simple":
            self._setup_simple_logging()
        else:
            raise ValueError(f"Неизвестный формат логов: {self.log_format}")

    def _setup_simple_logging(self):
        logger.add(
            self.full_log_path,
            format=self.FORMAT_SIMPLE,
            level="DEBUG",
            rotation="10 MB",
            compression="zip",
        )

    def _setup_json_logging(self):
        from utils.json_log_serializer import JsonLogSerializer

        serializer = JsonLogSerializer()

        def sink(message):
            record = message.record
            new_entry = json.loads(serializer.serialize(record))

            file_path = self.full_log_path

            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            else:
                logs = []

            logs.append(new_entry)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)

        logger.add(sink, level="DEBUG", backtrace=True, diagnose=True)

    def get_logger(self, context: str = None):
        if context:
            return logger.bind(context=context)
        return logger