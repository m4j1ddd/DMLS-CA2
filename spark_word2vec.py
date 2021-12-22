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
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    df_lines = []
    for line in lines:
        df_lines.append((line.split(" "),))
    df = spark.createDataFrame(df_lines, ["line"])
    df.show(8)

    word2vec = Word2Vec(vectorSize=5, seed=42, inputCol="line", outputCol="feature", numPartitions=partitions)
    model = word2vec.fit(df)

    model.getVectors().show()
    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.write().overwrite().save('Word2Vec.Model')
    spark.stop()
