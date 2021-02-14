from numpy import median, vectorize, multiply
from scipy.stats import mode

def subtract_median(depth, instance):
    return depth - median(depth)

def dubtract_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    median = median(depth[instance == instance_num])
    maskval = (max(depth[instance == instance_num]) - median) * 1.1
    return vectorize(lambda d, i: d - median if i == instance_num else maskval)

def divide_median(depth, instance):
    return depth / median(depth)

def divide_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    return multiply(depth, instance == instance_num) / median(depth[instance == instance_num])

def rescale_minmax(depth, instance):
    instance_num = mode(instance[instance != 0])[0][0]
    car_depth = depth[instance == instance_num]
    minval, maxval = min(car_depth), max(car_depth)
    return vectorize(lambda x, y: (x-minval)/(maxval-minval) if y == instance_num else 1.1)(depth, instance)

def _get_instance_num(instance):
    return mode(instance[instance != 0])[0][0]
