import random
import time

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

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("PythonSort-Abdollahi") \
        .getOrCreate()

    generated_list = list(generate_list(500000))

    sorted_list = execute_merge_sort(generated_list)

    sc = spark.sparkContext

    # rdd = sc.parallelize(generate_list).mapPartitions(execute_merge_sort).collect()
    sc.parallelize(generated_list).flatMap(lambda x: x).glom().mapPartitions(execute_merge_sort).collect()
