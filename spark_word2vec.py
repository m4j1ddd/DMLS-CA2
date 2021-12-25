import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2Vec-Abdollahi") \
        .getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    try:
        f_path = "/home/shared_files/CA2/wiki_corpus"
        f = open(f_path, 'r')
    except OSError:
        try:
            f_path = "wiki_corpus"
            f = open(f_path, 'r')
        except OSError:
            print("Could not read file")
            sys.exit()
    f.close()

    data = spark.sparkContext.textFile(f_path).map(lambda line: line.split(' ')).map(lambda arr: (arr,))
    df = spark.createDataFrame(data, ["text"])
    df.show(10)

    word2vec = Word2Vec(vectorSize=16, numPartitions=partitions, inputCol="text", outputCol="result")
    model = word2vec.fit(df)
    model.getVectors().show()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.write().overwrite().save('hdfs://raspberrypi-dml0:9000/abdollahi/Word2Vec.Model')

    spark.stop()
