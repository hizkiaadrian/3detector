from CarGenerator import CarGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Car import calculate_score

kitti_path = '/scratch/local/hdd/hizkia/kitti'

def calculate_EM_score(normalized_samples, cov):
    inv_cov = np.linalg.inv(cov)
    return sum(np.apply_along_axis(lambda x: x.T @ inv_cov @ x, 0, normalized_samples))

generator = CarGenerator(kitti_path, date=None, reference_rectangle=(64, 128), min_original_rectangle=(16,32), mean=None, cov=None).load_dataset()
D = np.array([x.normalized_depth.ravel() for x in generator]).T

mean = D.mean(axis = 1)
A = D - mean.reshape((-1,1))
cov = np.cov(A)

score = calculate_EM_score(A, cov)
print(f"Initializing EM yields a score of {score}...")

num_of_iters = 10
for iter_num in range(len(num_of_iters)):
    generator = CarGenerator(kitti_path, date=None, reference_rectangle=(64, 128), min_original_rectangle=(16,32), mean=mean, cov=cov).load_dataset()
    D = np.array([x.normalized_depth.ravel() for x in generator]).T

    temp_mean = D.mean(axis = 1)
    A = D - temp_mean.reshape((-1,1))
    temp_cov = np.cov(A)

    temp_score = calculate_EM_score(A, temp_cov)
    if temp_score > score:
        break

    score = temp_score
    mean = temp_mean
    cov = temp_cov
    print(f"EM iteration {iter_num} yields a score of {score}...")
    
print(f"EM stopped with a score of {score}. Saving data...")

np.savez("/scratch/local/hdd/hizkia/data.npz", mean=mean, cov=cov, inv_cov = np.linalg.inv(cov))

# print("Data saved successfully")
