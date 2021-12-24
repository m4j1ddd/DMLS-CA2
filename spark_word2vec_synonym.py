from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import SparkSession

from pyspark.sql.functions import format_number as fmt

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model = Word2VecModel.load('hdfs://raspberrypi-dml0:9000/abdollahi/Word2Vec.Model')
    print("iran synonyms: ")
    model.findSynonyms("iran", 10).select("word", (1 - fmt("similarity", 5)).alias("distance")).show()
    print("tehran synonyms: ")
    model.findSynonyms("tehran", 10).select("word", (1 - fmt("similarity", 5)).alias("distance")).show()
    print("learning synonyms: ")
    model.findSynonyms("learning", 10).select("word", (1 - fmt("similarity", 5)).alias("distance")).show()
    print("science synonyms: ")
    model.findSynonyms("science", 10).select("word", (1 - fmt("similarity", 5)).alias("distance")).show()

    v = model.getVectors()
    v.show()
    v_king = v.select(v.vector).where(v.word == 'king').collect()[0].vector
    print("king = " + str(v_king))
    v_man = v.select(v.vector).where(v.word == 'man').collect()[0].vector
    print("man = " + str(v_man))
    v_woman = v.select(v.vector).where(v.word == 'woman').collect()[0].vector
    print("woman = " + str(v_woman))
    v_queen = v.select(v.vector).where(v.word == 'queen').collect()[0].vector
    print("queen = " + str(v_queen))
    v_left = v_king - v_man + v_woman
    print("king - man + woman = " + str(v_left))
    print("length of sample vector: " + str(len(v_left)))

    spark.stop()
