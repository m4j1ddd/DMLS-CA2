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
    content = f.read()
    f.close()

    spark = SparkSession \
        .builder \
        .appName("Word2Vec-Abdollahi") \
        .getOrCreate()

    documentDF = spark.createDataFrame([(content.split(" "),)], ["text"])

    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

    spark.stop()
