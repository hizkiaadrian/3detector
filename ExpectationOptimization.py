from CarGenerator import CarGenerator, Direction
import numpy as np
import matplotlib.pyplot as plt

def calculate_EM_score(normalized_samples, cov):
    inv_cov = np.linalg.inv(cov)
    return sum(np.apply_along_axis(lambda x: x.T @ inv_cov @ x, 0, normalized_samples))

def _help(i, x):
    print(i,end='\r')
    return x.normalized_depth.ravel()

class ExpectationMaximization:
    def __init__(
        self, 
        base_path, 
        date=None, 
        reference_rectangle=(64,128), 
        min_original_rectangle=(32,64), 
        max_iters=10):

        self.base_path = base_path
        self.date = date
        self.reference_rectangle=reference_rectangle
        self.min_original_rectangle=min_original_rectangle
        self.max_iters = max_iters

    def run(self):
        generator = CarGenerator(
            self.base_path, 
            date=self.date, 
            reference_rectangle=self.reference_rectangle, 
            min_original_rectangle=self.min_original_rectangle, 
            mean=None, 
            cov=None).load_dataset()
        print("Generating samples...")
        D = np.array([_help(i,x) for i,x in enumerate(generator)]).T

        mean = D.mean(axis = 1)
        A = D - mean.reshape((-1,1))
        cov = np.cov(A)

        score = calculate_EM_score(A, cov)
        print(f"Initializing EM yields a score of {score}...")

        for iter_num in range(self.max_iters):
            optimize_direction = Direction(iter_num % 2)
            generator = CarGenerator(
                self.base_path,
                date=self.date,
                reference_rectangle=self.reference_rectangle,
                min_original_rectangle=self.min_original_rectangle,
                mean=mean,
                cov=cov,
                optimize_direction=optimize_direction
            )

            temp_D = np.array([_help(i,x) for i,x in enumerate(generator)]).T

            temp_mean = temp_D.mean(axis=1)
            temp_A = temp_D - temp_mean.reshape((-1,1))
            temp_cov = np.cov(temp_A)

            temp_score = calculate_EM_score(temp_A, temp_cov)
            if temp_score < score:
                D = temp_D
                mean = temp_mean
                cov = temp_cov
                score = temp_score
                print(f"EM iteration {iter_num} yields a score of {score}...")
            else:
                break

        print(f"EM stopped with a score of {score}")
        return D, mean, cov
