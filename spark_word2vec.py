import sys

from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

if __name__ == "__main__":
    try:
        f = open("/home/shared_files/CA2/wiki_corpus", 'r')
    except OSError:
        try:
            f = open("wiki_corpus", 'r')
        except OSError:
            print("Could not read file")
            sys.exit()
    lines = f.read().splitlines()
    f.close()

    spark = SparkSession \
        .builder \
        .appName("Word2Vec-Abdollahi") \
        .getOrCreate()

    df_lines = []
    for line in lines[:5]:
        df_lines.append((line.split(" "),))
    documentDF = spark.createDataFrame(df_lines, ["text"])

    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)
    model.getVectors().show()
    word2Vec.save()
    spark.stop()
