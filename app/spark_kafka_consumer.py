import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, struct, to_json, lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, ArrayType
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline, PipelineModel
# from pyspark.ml.feature import IndexToString #new
import json
import os
import time
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkKafkaConsumer:
    def __init__(self):
        self.spark = self._create_spark_session()
        self.schema = self._get_schema()
        # self.nlp_pipeline = self._create_nlp_pipeline()
        self.classifier_model = PipelineModel.load("../data/models/token_classifier_nlp_cleaned")

    def _create_spark_session(self):
        logger.info("Creating Spark session...")
        return (SparkSession.builder
                .appName("Kafka-Spark-NLP-Pipeline")
                .config("spark.jars.packages",
                        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.7.3") # new
                .config("spark.driver.memory", "2g")
                .config("spark.executor.memory", "2g")
                .config("spark.sql.streaming.checkpointLocation", "/data/checkpoint")
                .getOrCreate())

    def _get_schema(self):
        return StructType([
            StructField("PostID", StringType(), True),
            StructField("PostTelegramID", StringType(), True),
            StructField("PostText", StringType(), True),
            StructField("PostViewCount", IntegerType(), True),
            StructField("PostCreatedAt", StringType(), True),
            StructField("PostEditedAt", StringType(), True),
            StructField("PostParsedAt", StringType(), True),
            StructField("ChannelID", StringType(), True),
            StructField("ChannelParticipantsCount", IntegerType(), True),
            StructField("ChannelMainUsername", StringType(), True),
            StructField("ChannelTitle", StringType(), True)
        ])


    def is_stream_enabled(self):
        try:
            df = self.spark.read \
                .format("jdbc") \
                .option("url", "jdbc:postgresql://db:5432/streamdb") \
                .option("dbtable", "django_app_streamstatus") \
                .option("user", "streamuser") \
                .option("password", "streampass") \
                .option("driver", "org.postgresql.Driver") \
                .load()

            row = df.orderBy(col("id").desc()).limit(1).collect()[0]
            return row["is_active"]
        except Exception as e:
            logger.error(f"Error reading stream control: {e}")
            return False

    @staticmethod
    def write_to_json_file(batch_df, batch_id):
        data = batch_df.select(to_json(col("classified_data"))).rdd.map(lambda r: r[0]).collect()
        with open("/data/all_messages.json", "a", encoding="utf-8") as f:
            for json_line in data:
                f.write(json_line + "\n")

        with open("/data/stream_metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)

        with open("/data/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        throughput_limit = config.get("throughput_limit", 1)
        now = datetime.datetime.utcnow()
        prev_time = datetime.datetime.fromisoformat(metrics.get("last_update", datetime.datetime.utcnow()))
        elapsed = (now - prev_time).total_seconds() if prev_time else 0
        current_throughput = len(data) / elapsed if elapsed > 0 else 0

        metrics["current_throughput"] = round(current_throughput, 2)
        metrics["throughput_limit"] = throughput_limit
        metrics["status"] = "ok" if current_throughput >= throughput_limit else "slow"

        metrics['received_count'] += len(data)
        metrics['classified_count'] += len(data)
        metrics['last_update'] = datetime.datetime.utcnow().isoformat()

        metrics["uptime_seconds"] = int(
            (now - datetime.datetime.fromisoformat(metrics["start_time"])).total_seconds())

        with open("/data/stream_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        log_entry = {
            "timestamp": now.isoformat(),
            "received": metrics["received_count"],
            "classified": metrics["classified_count"],
            "throughput": metrics["current_throughput"],
            "limit": metrics["throughput_limit"],
            "uptime": metrics["uptime_seconds"],
            "status": metrics["status"]
        }

        log_history = []

        with open("/data/performance_logs.json", "r", encoding="utf-8") as f:
            log_history = json.load(f)

        log_history.append(log_entry)

        with open("/data/performance_logs.json", "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2)

    @staticmethod
    def write_to_postgres(batch_df, batch_id):
        enriched_df = batch_df.select(
            col("classified_data.PostID").alias("post_id"),
            col("classified_data.PostText").alias("post_text"),
            to_json(col("classified_data.tokens")).alias("tokens"),
            col("classified_data.ChannelTitle").alias("channel_title"),
            col("classified_data.ChannelID").alias("channel_id"),
            col("classified_data.PostViewCount").alias("post_view_count"),
            col("classified_data.PostCreatedAt").cast("timestamp").alias("post_created_at"),
            col("classified_data.category").alias("category"),
            col("classified_data.status").alias("status"),
            col("classified_data.processing_level").alias("processing_level")
        ).withColumn("parsed_at", current_timestamp())

        enriched_df.write \
            .format("jdbc") \
            .option("url", "jdbc:postgresql://db:5432/streamdb") \
            .option("dbtable", "django_app_classifiedmessage") \
            .option("user", "streamuser") \
            .option("password", "streampass") \
            .option("driver", "org.postgresql.Driver") \
            .mode("append") \
            .save()

    def process_stream(self):

        logger.info("Starting to process Kafka stream...")

        kafka_df = self.spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "kafka:9092") \
            .option("subscribe", "telegram-posts") \
            .option("startingOffsets", "latest") \
            .option("kafkaConsumer.timeout.ms", "10000") \
            .option("failOnDataLoss", "false") \
            .load()

        parsed_df = kafka_df.select(
            from_json(col("value").cast("string"), self.schema).alias("data")
        ).select("data.*")


        with open("/data/config.json", "r") as f:
            config = json.load(f)
        max_len = config.get("max_len", 100000)

        parsed_df = parsed_df.filter(col("PostText").isNotNull() & (col("PostText").rlike(f"^.{{0,{max_len}}}$")))

        parsed_df = parsed_df.withColumnRenamed("PostText", "text")
        processed_df = self.classifier_model.transform(parsed_df)

        classified_df = processed_df.withColumn(
            "classified_data",
            struct(
                col("PostID"),
                col("text").alias("PostText"), # neeeeeeeew
                col("finished_tokens").cast(ArrayType(StringType())).alias("tokens"),  #
                col("ChannelTitle"),
                col("ChannelID"),
                col("PostViewCount"),
                col("PostCreatedAt"),
                col("prediction").cast("int").alias("category"),  #  new
                #col("category_2"), # new
                lit("nlp_processed").alias("status"),
                lit("classified").alias("processing_level")
            )
        )

        query = classified_df.writeStream \
            .foreachBatch(self.write_to_json_file) \
            .outputMode("append") \
            .trigger(processingTime='30 seconds') \
            .start()


        console_query = processed_df.select("PostID", col("text").alias("PostText"), col("tokens.result")) \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .trigger(processingTime='30 seconds') \
            .start()

        return query, console_query

    def reset_metrics(self):
        initial_metrics = {
            "received_count": 0,
            "classified_count": 0,
            "start_time": datetime.datetime.utcnow().isoformat(),
            "last_update": datetime.datetime.utcnow().isoformat()
        }
        with open("/data/stream_metrics.json", "w", encoding="utf-8") as f:
            json.dump(initial_metrics, f, indent=2)


if __name__ == "__main__":
    consumer = SparkKafkaConsumer()

    consumer.reset_metrics()
    try:
        while True:
            if consumer.is_stream_enabled():
                logger.info("Stream is enabled. Starting processing...")

                consumer.reset_metrics()

                query, console_query = consumer.process_stream()
                query.awaitTermination()
                console_query.awaitTermination()

                logger.info("Stream terminated. Will re-check control flag in 10 seconds.")
                time.sleep(10)
            else:
                logger.info("Stream is disabled. Waiting 10 seconds before rechecking...")
                time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Terminated by user.")
    finally:
        try:
            query.stop()
            console_query.stop()
        except:
            pass
        consumer.spark.stop()
