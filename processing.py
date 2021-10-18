import multiprocessing as mp
import queue
import random


def my_func(x):
    print(mp.current_process())
    print(x**x)

def main():
    print("Number of CPUs:", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(my_func, [4,2,3,5,3,2,1,2])
    result_set_2 = pool.map(my_func, [4,6,5,4,6,3,23,4,6])

    print(result)
    print(result_set_2)

if __name__ == "__main__":
    main()