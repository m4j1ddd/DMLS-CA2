import sys

from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import SparkSession

from pyspark.sql.functions import format_number as fmt

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model = Word2VecModel.load('Word2Vec.Model')
    model.findSynonyms("iran", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    model.findSynonyms("tehran", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    model.findSynonyms("learning", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    model.findSynonyms("science", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()

    v = model.getVectors()
    v.show()
    left = v.select(v.vector).where(v.word == 'king').collect()[0].vector - v.select(v.vector).where(v.word == 'man').collect()[0].vector + v.select(v.vector).where(v.word == 'woman').collect()[0].vector
    right = v.select(v.vector).where(v.word == 'queen').collect()[0].vector
    print("king - man + woman = " + str(left))
    print("queen = " + str(right))

    spark.stop()
