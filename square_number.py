import time
import numpy as np


result = []

# Fx. to get the square of the number...
def square_number(no):
    return (no*no)


# Fx. to compute square of a range of a number...
def get_square_range(start_no, end_no):
    for i in np.arange(start_no, end_no):
        time.sleep(1)
        result.append(square_number(i))

    return result


start = time.time()
final_result = get_square_range(1, 21)
end = time.time()
print('\n The fx took {:.2f} s to compute.'.format(end - start))
print(final_result)
