from typing import final
import numpy as np
import time
from joblib import Memory

from square_number import square_number

# Define a location to store cache...
location = 'temp/cache_dir'
memory = Memory(location, verbose=0)

memory.clear(warn=False)
