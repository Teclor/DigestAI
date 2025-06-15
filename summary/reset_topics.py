from kafka import KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import UnknownTopicOrPartitionError, NodeNotReadyError, NoBrokersAvailable
import time

BOOTSTRAP_SERVERS = "kafka:9092"
PARTITIONS = 4
REPLICATION_FACTOR = 1


def connect_with_retries(retries=15, delay=5):
    for i in range(retries):
        try:
            admin = KafkaAdminClient(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                client_id="kafka_reset_script"
            )
            print("✅ Успешно подключились к Kafka")
            return admin
        except (NodeNotReadyError, NoBrokersAvailable) as e:
            print(f"❌ Kafka ещё не готова, попытка {i + 1} из {retries}. Жду {delay} сек... Ошибка: {e}")
            time.sleep(delay)
    raise ConnectionError("Не удалось подключиться к Kafka после нескольких попыток")


def reset_all_topics(admin_client):
    try:
        topics = admin_client.list_topics()
        print(f"Найдено топиков: {topics}")

        for topic in topics:
            try:
                print(f"Удаляю топик: {topic}")
                admin_client.delete_topics([topic])
            except UnknownTopicOrPartitionError:
                print(f"Топик {topic} не существует")

        print("Все топики удалены.")
    except Exception as e:
        print(f"Ошибка при работе с топиками: {e}")


def create_topic(admin_client, topic_name, partitions=PARTITIONS, replication_factor=REPLICATION_FACTOR):
    try:
        topic = NewTopic(
            name=topic_name,
            num_partitions=partitions,
            replication_factor=replication_factor
        )
        admin_client.create_topics([topic])
        print(f"Создан топик: {topic_name}")
    except Exception as e:
        print(f"Ошибка при создании топика '{topic_name}': {e}")


if __name__ == "__main__":
    try:
        admin = connect_with_retries()
        reset_all_topics(admin)

        admin.close()
    except Exception as e:
        print(f"Ошибка: {e}")