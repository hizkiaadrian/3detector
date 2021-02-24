from numpy import median, vectorize, multiply
from scipy.stats import mode

def subtract_median(depth, instance):
    return depth - median(depth)

def subtract_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    depth_median = median(depth[instance == instance_num])
    return vectorize(lambda d, i: d - depth_median if i == instance_num else -1)(depth, instance)

def divide_median(depth, instance):
    return depth / median(depth)

def divide_median_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    depth_median = median(depth[instance == instance_num])
    return vectorize(lambda d, i: d - depth_median if i == instance_num else -1)(depth, instance)

def rescale_minmax(depth, instance):
    return (depth - depth.min()) / (depth.max() - depth.min())

def rescale_minmax_masked(depth, instance):
    instance_num = _get_instance_num(instance)
    car_depth = depth[instance == instance_num]
    minval, maxval = min(car_depth), max(car_depth)
    return vectorize(lambda x, y: (x-minval)/(maxval-minval) if y == instance_num else -1)(depth, instance)

def _get_instance_num(instance):
    return mode(instance[instance != 0])[0][0]
