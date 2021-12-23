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
    model.getVectors().show()
    v = model.getVectors()
    v.select(v.vector).where(v.word == 'king').show()
    v.select(v.vector).where(v.word == 'man').show()
    v.select(v.vector).where(v.word == 'woman').show()
    v.select(v.vector).where(v.word == 'queen').show()
    # model.findSynonyms("iran", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    # model.findSynonyms("tehran", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    # model.findSynonyms("learning", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    # model.findSynonyms("science", 10).select("word", (1 - fmt("similarity", 100)).alias("distance")).show()
    #
    # documentDF = spark.createDataFrame([
    #     ("king",),
    #     ("man",),
    #     ("woman",),
    #     ("queen",)
    # ], ["text"])
    # # Learn a mapping from words to Vectors.
    # # word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    # # model = word2Vec.fit(documentDF)
    # w2v = Word2Vec.load('Word2Vec.Model')
    # result = w2v.transform(documentDF)

    spark.stop()
