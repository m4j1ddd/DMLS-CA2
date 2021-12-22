import sys

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
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
    for line in lines:
        df_lines.append((line.split(" "),))
    df = spark.createDataFrame(df_lines, ["text"])
    df.show(8)
    word2vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="feature")
    model = word2vec.fit(df)
    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.save('Word2Vec.Model')
    spark.stop()
