from messages.kafka_chat_producer import KafkaChatProducer

if __name__ == "__main__":
    k_producer = KafkaChatProducer(default_resource_dir="resources/chats")
    k_producer.send_from_folder()
    k_producer.close()