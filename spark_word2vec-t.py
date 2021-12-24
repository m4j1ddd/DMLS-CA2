import sys

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec, Tokenizer

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2Vec-Abdollahi") \
        .getOrCreate()
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

    df = spark.read.load(f_path, format="csv", inferSchema="true")
    df.show(10)

    tokenizer = Tokenizer(inputCol="_c0", outputCol="word")
    word2Vec = Word2Vec(vectorSize=60, inputCol="word", outputCol="feature")
    pipeline = Pipeline(stages=[tokenizer, word2Vec])

    p_model = pipeline.fit(df)
    model = p_model.stages[1]
    model.getVectors().show()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.write().overwrite().save('Word2Vec.Model')

    spark.stop()
