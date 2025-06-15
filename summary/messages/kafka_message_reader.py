from typing import List, Dict, Optional, Tuple
from kafka import KafkaConsumer
import json
import os
from config import config


class KafkaMessageReader:
    def __init__(self, bootstrap_servers: str = config.kafka.bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers
        self.topic_map = self._load_topic_map()

    def _load_topic_map(self) -> Dict[str, str]:
        map_path = os.path.join("resources", "chats", "topic_map.json")
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def read_messages(self, topic: str, limit: int = 10) -> List[Dict]:
        if not 1 <= limit <= config.kafka.max_messages_to_read:
            raise ValueError(
                f"Limit должен быть между 1 и {config.kafka.max_messages_to_read}"
            )

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap_servers,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            group_id=None,
            fetch_max_bytes=config.kafka.fetch_max_bytes,
            max_partition_fetch_bytes=config.kafka.max_partition_fetch_bytes,
            max_poll_records=min(limit, config.kafka.max_poll_records),
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        messages = []
        try:
            while len(messages) < limit:
                records = consumer.poll(timeout_ms=config.kafka.poll_timeout_ms)
                if not records:
                    break
                for msgs in records.values():
                    for msg in msgs:
                        messages.append(msg.value)
                        if len(messages) >= limit:
                            break
                    if len(messages) >= limit:
                        break
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении сообщений из Kafka: {e}")
        finally:
            consumer.close()

        return messages

    def get_chat_info_and_messages(self, topic: str, limit: int = 10) -> Tuple[Optional[str], Optional[str], List[Dict]]:
        chat_id = topic
        chat_name = self.topic_map.get(topic)
        messages = self.read_messages(topic, limit)
        return chat_id, chat_name, messages

    @staticmethod
    def format_messages_for_summary(messages: List[Dict]) -> str:
        result = []
        for msg in messages:
            data = msg.get("data", {})
            if isinstance(data, dict) and "text" in data:
                result.append(data["text"])
        return "\n".join(result)