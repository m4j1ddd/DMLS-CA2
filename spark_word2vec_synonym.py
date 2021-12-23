import sys

from pyspark.ml import PipelineModel
from pyspark.ml.feature import Word2VecModel
from pyspark.sql import SparkSession

from pyspark.sql.functions import format_number as fmt

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model = Word2VecModel.load('LocalWord2Vec.Model')
    model.getVectors().show()

    model.findSynonyms("iran", 10).select("word", (1 - fmt("similarity", 40)).alias("distance")).show()
    model.findSynonyms("tehran", 10).select("word", (1 - fmt("similarity", 40)).alias("distance")).show()
    model.findSynonyms("learning", 10).select("word", (1 - fmt("similarity", 40)).alias("distance")).show()
    model.findSynonyms("science", 10).select("word", (1 - fmt("similarity", 40)).alias("distance")).show()

    spark.stop()
