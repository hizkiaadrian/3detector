from Preprocessor.ExpectationMaximization import ExpectationMaximization
from Preprocessor.DepthNormalization import divide_median_masked

kitti_path = "/scratch/local/hdd/hizkia/kitti"
reference_rectangle = (64, 128)
min_original_rectangle = (32, 64)
depth_normalization_func = divide_median_masked

expectationMaximizer = ExpectationMaximization(
    base_path=kitti_path,
    reference_rectangle=reference_rectangle,
    min_original_rectangle=min_original_rectangle,
    depth_normalization_func=depth_normalization_func
)

# expectationMaximizer.run()

dataset_save_folder = "/scratch/local/hdd/hizkia/cars"

expectationMaximizer.save(dataset_save_folder=dataset_save_folder)
