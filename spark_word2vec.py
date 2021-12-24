# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:15:45 2021

@author: ALBALOO
"""
import sys
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import format_number as fmt

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

    data = spark.sparkContext.textFile(f_path).map(lambda line: line.split(' ')).map(lambda arr: (arr,))
    df = spark.createDataFrame(data, ["text"])
    df.show(10)

    word2vec = Word2Vec(inputCol="text", outputCol="feature")
    model = word2vec.fit(df)
    model.getVectors().show()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.write().overwrite().save('Word2Vec.Model')

    spark.stop()
