import random
import sys
import time
from array import array

from pyspark.sql import SparkSession


def execute_merge_sort(generated_list):
    start_time = time.time()
    sorted_list = merge_sort(generated_list)
    elapsed = time.time() - start_time
    print('Simple merge sort: %f sec' % elapsed)
    return sorted_list


def generate_list(length):
    N = length
    generated_list = [random.random() for num in range(N)]
    return generated_list


def merging(left_side, right_side):
    result = []
    i = j = 0
    while i < len(left_side) and j < len(right_side):
        if left_side[i] <= right_side[j]:
            result.append(left_side[i])
            i += 1
        else:
            result.append(right_side[j])
            j += 1
    if i == len(left_side):
        result.extend(right_side[j:])
    else:
        result.extend(left_side[i:])
    return result


def merge_sort(generated_list):
    print(generated_list)
    if len(generated_list) <= 1:
        return generated_list
    middle_value = len(generated_list) // 2
    sorted_list = merging(merge_sort(generated_list[:middle_value]), merge_sort(generated_list[middle_value:]))
    return sorted_list


def is_sorted(num_array):
    for i in range(1, len(num_array)):
        if num_array[i] < num_array[i - 1]:
            return False
    return True


def merge(x, y):
    if isinstance(x, list):
        left_side = x
    else:
        left_side = [x]

    if isinstance(y, list):
        right_side = y
    else:
        right_side = [y]

    return merging(left_side, right_side)


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
