import sys

from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.sql import SparkSession

import findspark
findspark.add_packages('mysql:mysql-connector-java:8.0.11')

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("Word2VecSynonym-Abdollahi") \
        .getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    # hdfs://raspberrypi-dml0:9000/abdollahi/
    model = Word2VecModel.load('Word2Vec.Model')
    model.getVectors().show()

    spark.stop()
