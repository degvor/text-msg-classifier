import json
import random
import time
from datetime import datetime, timedelta
from kafka import KafkaProducer
from faker import Faker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()


class TelegramDataGenerator:
    def __init__(self):
        self.channels = [
            {"id": "channel1", "title": "Tech News", "username": "technews", "participants": 15000},
            {"id": "channel2", "title": "Global Politics", "username": "globalpolitics", "participants": 8000},
            {"id": "channel3", "title": "Science Updates", "username": "sciencenews", "participants": 12000},
            {"id": "channel4", "title": "Entertainment Buzz", "username": "entertainment", "participants": 20000},
            {"id": "channel5", "title": "Business Daily", "username": "businessdaily", "participants": 10000}
        ]

        self.post_topics = {
            "Tech News": [
                "New AI breakthrough announced by leading tech company!",
                "Revolutionary quantum computing chip unveiled",
                "Major cryptocurrency exchange faces security breach",
                "Tech giant acquires startup for $2 billion",
                "Apple announces new product line for 2024"
            ],
            "Global Politics": [
                "UN Security Council holds emergency meeting",
                "International trade tensions escalate",
                "Global climate agreement reaches milestone",
                "Election results surprise political analysts",
                "Diplomatic relations between nations improve"
            ],
            "Science Updates": [
                "Scientists discover new exoplanet in habitable zone",
                "Medical breakthrough: new cancer treatment shows promise",
                "Climate change impact more severe than predicted",
                "Space mission launches to explore asteroid",
                "Renewable energy efficiency reaches record high"
            ],
            "Entertainment Buzz": [
                "Hollywood star announces surprise marriage",
                "Record-breaking box office weekend for new blockbuster",
                "Music industry scandal rocks award ceremony",
                "Celebrity chef opens restaurant chain",
                "Film festival announces surprise winner"
            ],
            "Business Daily": [
                "Stock market shows unexpected volatility",
                "Tech stocks lead market rally",
                "Retail sector faces challenging quarter",
                "Corporate earnings exceed expectations",
                "Global supply chain disruptions continue"
            ]
        }

    def generate_telegram_post(self):
        channel = random.choice(self.channels)
        post_text = random.choice(self.post_topics[channel["title"]])

        base_views = int(channel["participants"] * random.uniform(0.01, 0.15))
        view_count = base_views + random.randint(-100, 500)

        created_at = datetime.now() - timedelta(minutes=random.randint(0, 1440))
        edited_at = None
        if random.random() < 0.3:
            edited_at = created_at + timedelta(minutes=random.randint(1, 60))

        post_data = {
            "PostID": fake.uuid4(),
            "PostTelegramID": str(random.randint(1000000, 9999999)),
            "PostText": post_text,
            "PostViewCount": view_count,
            "PostCreatedAt": created_at.isoformat(),
            "PostEditedAt": edited_at.isoformat() if edited_at else None,
            "PostParsedAt": datetime.now().isoformat(),
            "ChannelID": channel["id"],
            "ChannelParticipantsCount": channel["participants"],
            "ChannelMainUsername": channel["username"],
            "ChannelTitle": channel["title"]
        }

        return post_data


class KafkaDataProducer:
    def __init__(self, bootstrap_servers='kafka:9092', topic='telegram-posts'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.generator = TelegramDataGenerator()

    def connect(self):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return False

    def send_test_data(self, num_messages=100, delay=1):
        if not self.producer:
            if not self.connect():
                return

        try:
            for i in range(num_messages):
                post_data = self.generator.generate_telegram_post()
                self.producer.send(self.topic, value=post_data)
                logger.info(f"Sent message {i + 1}/{num_messages}: PostID={post_data['PostID']}")
                time.sleep(delay)

            self.producer.flush()
            logger.info("All messages sent successfully")
        except Exception as e:
            logger.error(f"Error sending messages: {e}")
        finally:
            self.producer.close()


if __name__ == "__main__":
    producer = KafkaDataProducer()

    while not producer.connect():
        logger.info("Waiting for Kafka to be ready...")
        time.sleep(5)

    while True:
        producer.send_test_data(num_messages=100, delay=2)
        logger.info("Waiting before sending next batch...")
        time.sleep(60)