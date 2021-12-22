import sys

from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession

from pyspark.sql.functions import format_number as fmt

def print_similar(word, model):
    synonyms = model.findSynonymsArray(word, 10)
    print(word + " is similar to: ")
    for synonym in synonyms:
        print(synonym[0])


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model = Word2VecModel.load('Word2Vec.Model')
    model.getVectors().show()

    print_similar("iran", model)
    print_similar("tehran", model)
    print_similar("learning", model)
    print_similar("science", model)

    # model.findSynonyms("could", 2).select("word", fmt("similarity", 5).alias("similarity")).show()

    spark.stop()
