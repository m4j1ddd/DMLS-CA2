import sys
from array import array

from pyspark.sql import SparkSession


def print_in_file(arr, file_name):
    f = open(file_name, 'w+')
    for i in range(len(arr)):
        f.write(str(arr[i]))
        f.write('\n')
    f.close()


if __name__ == "__main__":
    try:
        f = open("/home/shared_files/CA1/to_sort.txt", 'r')
    except OSError:
        try:
            f = open("to_sort.txt", 'r')
        except OSError:
            print("Could not read file")
            sys.exit()
    arr = array('I', [0]) * 2000000
    count = 0
    for each in f:
        arr[count] = int(each)
        count += 1
    f.close()

    spark = SparkSession \
        .builder \
        .appName("PythonSort-Abdollahi") \
        .getOrCreate()
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rdd = spark.sparkContext.parallelize(arr, partitions).map(lambda x: (x, 1)).sortByKey()
    print("Number of partitions: " + str(rdd.getNumPartitions()))
    output = rdd.collect()
    sorted_arr = []
    for (num, unitcount) in output:
        sorted_arr.append(num)
    spark.stop()

    print_in_file(sorted_arr, "sorted.txt")
