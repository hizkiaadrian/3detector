from Preprocessor.CarGenerator import CarGenerator
from Preprocessor.BoxOperations import Direction
from Preprocessor.DepthNormalization import divide_median
from numpy import apply_along_axis, array, savez
import numpy as np
from numpy.linalg import inv
from pickle import dump, HIGHEST_PROTOCOL
from os import mkdir
from os.path import exists

def calculate_EM_score(normalized_samples, cov):
    inv_cov = inv(cov)
    return sum(apply_along_axis(lambda x: x.T @ inv_cov @ x, 0, normalized_samples))

class ExpectationMaximization:
    def __init__(
        self, 
        base_path, 
        dates=None, 
        reference_rectangle=(64,128), 
        min_original_rectangle=(32,64),
        depth_normalization_func=None, 
        max_iters=10):
        self.base_path = base_path
        self.dates = dates
        self.reference_rectangle = reference_rectangle
        self.min_original_rectangle = min_original_rectangle
        self.depth_normalization_func= depth_normalization_func if depth_normalization_func is not None else divide_median
        self.max_iters = max_iters
        self.__result = {"mean":None, "cov":None, "optimize_direction": None}

    def run(self):
        def _help(i, x):
            print(i,end='\r')
            return x.normalized_depth.ravel()

        generator = CarGenerator(
            self.base_path, 
            dates = self.dates, 
            reference_rectangle = self.reference_rectangle, 
            min_original_rectangle = self.min_original_rectangle,
            depth_normalization_func = self.depth_normalization_func, 
            mean = None, 
            cov = None)
        print("Generating samples...")
        D = array([_help(i,x) for i,x in enumerate(generator)]).T

        if D.shape[1] < (self.reference_rectangle[0] * self.reference_rectangle[1]):
            raise ValueError("Number of samples is too little resulting in a singular covariance matrix")
        mean = D.mean(axis = 1)
        A = D - mean.reshape((-1,1))
        cov = np.cov(A)

        score = calculate_EM_score(A, cov)
        optimdir = None
        print(f"Initializing EM yields a score of {score}...")

        for iter_num in range(self.max_iters):
            optimize_direction = Direction(iter_num % 2 + 1)

            generator = CarGenerator(
                self.base_path,
                dates=self.dates,
                reference_rectangle=self.reference_rectangle,
                min_original_rectangle=self.min_original_rectangle,
                depth_normalization_func=self.depth_normalization_func,
                mean=mean,
                cov=cov,
                optimize_direction=optimize_direction
            )

            temp_D = array([_help(i,x) for i,x in enumerate(generator)]).T
            if temp_D.shape[1] < (self.reference_rectangle[0] * self.reference_rectangle[1]):
                raise ValueError("Number of samples is too little resulting in a singular covariance matrix")
            
            temp_mean = temp_D.mean(axis=1)
            temp_A = temp_D - temp_mean.reshape((-1,1))
            temp_cov = np.cov(temp_A)

            temp_score = calculate_EM_score(temp_A, temp_cov)
            print(f"EM iteration {iter_num} yields a score of {temp_score}")
            if temp_score < score:
                D = temp_D
                mean = temp_mean
                cov = temp_cov
                score = temp_score
                optimdir = optimize_direction
                self.__result = {"mean":mean, "cov":cov, "optimize_direction": optimdir}
            else:
                break

        print(f"EM stopped with a score of {score}")

    def get_result(self):
        return self.__result

    def save(self, dataset_save_folder):
        generator = CarGenerator(self.base_path,
                dates=self.dates,
                reference_rectangle=self.reference_rectangle,
                min_original_rectangle=self.min_original_rectangle,
                depth_normalization_func=self.depth_normalization_func,
                mean=self.__result['mean'],
                cov=self.__result['cov'],
                optimize_direction=self.__result['optimize_direction'])

        if not exists(dataset_save_folder):
            mkdir(dataset_save_folder)

        for i, car in enumerate(generator):
            with open(f"{dataset_save_folder}/{i}.pkl", 'wb') as file:
                dump(car, file, HIGHEST_PROTOCOL)
            print(f"Saving: {i}", end="\r")
