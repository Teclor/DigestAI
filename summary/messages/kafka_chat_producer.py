import os
import json
import glob

from kafka import KafkaProducer
from kafka.errors import KafkaError

from utils.logger_configurator import LoggerConfigurator

logger_config = LoggerConfigurator("python_kafka.json", log_format="json")
logger = logger_config.get_logger("KafkaChatProducer")


class KafkaChatProducer:
    def __init__(self, bootstrap_servers=None, default_resource_dir="resources/chats"):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        self.default_resource_dir = default_resource_dir
        self.producer = None

    def _init_producer(self):
        if not self.producer:
            try:
                env_server = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
                logger.info(f"Значение KAFKA_BOOTSTRAP_SERVERS: {env_server}")
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                    batch_size=16384,
                    linger_ms=5,
                    buffer_memory=33554432,
                    max_in_flight_requests_per_connection=5,
                    acks=1,
                )
                logger.info(f"KafkaProducer успешно инициализирован для {self.bootstrap_servers}")
            except Exception as e:
                logger.error(f"Ошибка при инициализации KafkaProducer: {e}", exc_info=True)
                raise  # важно пробросить ошибку выше, чтобы не продолжать выполнение с None

    def _on_send_success(self, topic_name):
        def success(record_metadata):
            logger.debug(
                f"Топик: {record_metadata.topic}, "
                f"Партиция: {record_metadata.partition}, "
                f"Offset: {record_metadata.offset}"
            )

        return success

    def _on_send_error(self, topic_name):
        def error(excp):
            logger.error(f"Ошибка при отправке сообщения в топик {topic_name}: {excp}")
            if isinstance(excp, KafkaError):
                logger.debug(f"Код ошибки: {excp.args}, Подробности: {excp.message}")

        return error

    def send_from_folder(self, resource_dir=None):
        self._init_producer()
        resource_dir = resource_dir or self.default_resource_dir

        files = glob.glob(os.path.join(resource_dir, "*_*.json"))

        if not files:
            logger.warning(f"Файлы не найдены в директории {resource_dir}")
            return

        for file_path in files:
            file_name = os.path.basename(file_path)
            topic_name = file_name.replace(".json", "")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, dict) and "messages" in data:
                        messages = data["messages"]
                    elif isinstance(data, list):
                        messages = data
                    else:
                        logger.warning(f"Неизвестный формат данных в файле {file_name}")
                        continue

                    self._send_messages(topic_name, messages)

            except json.JSONDecodeError as e:
                logger.error(f"Ошибка в JSON файле {file_path}: {e}")

        self.producer.flush()
        logger.info("Завершена отправка всех файлов из папки")

    def send_raw_json(self, topic_name, messages):
        self._init_producer()

        try:
            logger.info(f"Отправка в топик {topic_name}, количество сообщений: {len(messages)}")
            self._send_messages(topic_name, messages)
            self.producer.flush()

        except json.JSONDecodeError as e:
            logger.error(f"Ошибка разбора JSON: {e}")
        except Exception as e:
            logger.exception(f"Непредвиденная ошибка: {e}")

    def _send_messages(self, topic_name, messages):
        for message in messages:
            record = {"data": message}

            future = self.producer.send(topic_name, value=record)
            future.add_errback(self._on_send_error(topic_name))

    def close(self):
        if self.producer:
            self.producer.close()