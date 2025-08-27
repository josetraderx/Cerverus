class KafkaProducer:
    def __init__(self, brokers: list[str]):
        self.brokers = brokers

    def send(self, topic: str, key: bytes, value: bytes) -> None:
        print(f"Sending message to {topic}")
