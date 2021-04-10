from numpy import median, vectorize, multiply
from scipy.stats import mode

def subtract_median(depth, instance):
    return depth - median(depth)

def divide_median(depth, instance):
    return depth / median(depth)

def divide_median_masked(depth, instance):
    depth_median = median(depth[instance == 1])
    return vectorize(lambda d, i: d - depth_median if i == 1 else -1)(depth, instance)

def rescale_minmax(depth, instance):
    return (depth - depth.min()) / (depth.max() - depth.min())

def rescale_minmax_masked(depth, instance):
    car_depth = depth[instance == 1]
    minval, maxval = min(car_depth), max(car_depth)
    return vectorize(lambda x, y: (x-minval)/(maxval-minval) if y == 1 else -1)(depth, instance)
