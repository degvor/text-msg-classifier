FROM bitnami/spark:3.4.1

USER root

RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /
RUN pip3 install -r /requirements.txt

RUN mkdir -p /app /data

RUN curl -o /opt/bitnami/spark/jars/spark-nlp_2.12-4.4.0.jar https://repo1.maven.org/maven2/com/johnsnowlabs/nlp/spark-nlp_2.12/4.4.0/spark-nlp_2.12-4.4.0.jar

WORKDIR /app

COPY ./app /app

RUN chmod +x /app/spark_entrypoint.sh

CMD ["/app/spark_entrypoint.sh"]