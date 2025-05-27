from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import Tokenizer, Normalizer, StopWordsCleaner
from pyspark.sql.functions import col
import json

spark = SparkSession.builder \
    .appName("TokenClassifierTraining") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.3") \
    .getOrCreate()

df = spark.read.option("header", "true").csv("tokenized_posts_labeled.csv")
df = df.withColumnRenamed("PostText", "text").withColumnRenamed("label", "rawLabel")

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner() \
    .setInputCols(["normalized"]) \
    .setOutputCol("tokens") \
    .setCaseSensitive(False) #\
    #.setStopWords(["і", "в", "на", "що", "до", "з", "як", "у", "та", "це", "a", "an", "the", "and", "or", "in", "of", "to"])

finisher = Finisher() \
    .setInputCols(["tokens"]) \
    .setOutputCols(["finished_tokens"]) \
    .setCleanAnnotations(False)

# label_indexer = StringIndexer(inputCol="rawLabel", outputCol="label")
label_indexer = StringIndexer(inputCol="rawLabel", outputCol="label") #new
indexer_model = label_indexer.fit(df) #new

label_mapping = {int(i): lbl for i, lbl in enumerate(indexer_model.labels)}
with open("../data/models/label_mapping.json", "w") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

cv = CountVectorizer(inputCol="finished_tokens", outputCol="rawFeatures", vocabSize=2000, minDF=2)
idf = IDF(inputCol="rawFeatures", outputCol="features")

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[
    document_assembler,
    tokenizer,
    normalizer,
    stopwords_cleaner,
    finisher,
    indexer_model, #label_indexer, #new
    cv,
    idf,
    lr
])

model = pipeline.fit(df)

model.write().overwrite().save("../data/models/token_classifier_nlp_cleaned")

print("train done")
print()
spark.stop()
