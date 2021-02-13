import numpy as np
from scipy.stats import mode

def subtract_median(depth, instance):
    return depth - np.median(depth)

def dubtract_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    median = np.median(depth[instance == instance_num])
    maskval = (max(depth[instance == instance_num]) - median) * 1.1
    return np.vectorize(lambda d, i: d - median if i == instance_num else maskval)

def divide_median(depth, instance):
    return depth / np.median(depth)

def divide_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    return np.multiply(depth, instance == instance_num) / np.median(depth[instance == instance_num])

def rescale_minmax(depth, instance):
    instance_num = mode(instance[instance != 0])[0][0]
    car_depth = depth[instance == instance_num]
    minval, maxval = min(car_depth), max(car_depth)
    return np.vectorize(lambda x, y: (x-minval)/(maxval-minval) if y == instance_num else 1.1)(depth, instance)

def _get_instance_num(instance):
    return mode(instance[instance != 0])[0][0]