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

    df = spark.read.load("wiki_corpus", format="csv", inferSchema="true")
    df.show(8)

    tokenizer = Tokenizer(inputCol="_c0", outputCol="word")
    word2Vec = Word2Vec(inputCol="word", outputCol="feature")
    pipeline = Pipeline(stages=[tokenizer, word2Vec])

    model = pipeline.fit(df)
    my_word2vec = model.stages[1]
    my_word2vec.getVectors().show()

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model.write().overwrite().save('Word2Vec.Model')

    spark.stop()
