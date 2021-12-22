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
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()

    df_lines = []
    for line in lines[:5]:
        df_lines.append((line.split(" "),))
    documentDF = spark.createDataFrame(df_lines, ["text"])
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    word2Vec = Word2Vec(vectorSize=3, minCount=0, numPartitions=partitions, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)
    model.getVectors().show()
    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
    word2Vec.save("hdfs://raspberrypi-dml0:9000/abdollahi/Word2Vec.Model")
    spark.stop()
