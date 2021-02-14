from BundleGenerator import BundleGenerator
from Car import Car
from os import listdir
from cv2 import resize, INTER_NEAREST
from numpy import array, zeros, block, median
from numpy.linalg import inv
from math import ceil
import BoxOperations

class CarGenerator:
    def __init__(self, 
                 base_dir, 
                 dates = None, 
                 reference_rectangle = (64, 128), 
                 min_original_rectangle = (32,64), 
                 depth_normalization_func=None, 
                 mean = None, 
                 cov = None, 
                 optimize_direction = BoxOperations.Direction.ALL
                ):
        if dates is not None and not all([date in listdir(base_dir) for date in dates]):
            raise ValueError
        self.bundle_generator = BundleGenerator(base_dir, dates)
        self.reference_rectangle = reference_rectangle
        self.min_original_rectangle = min_original_rectangle
        self.depth_normalization_func = depth_normalization_func
        self.mean = mean
        self.cov = cov
        self.inv_cov = inv(cov) if cov is not None else None
        self.optimize_direction = optimize_direction
        self.__internal_generator = self.__load_dataset()
    
    def __next__(self):
        return next(self.__internal_generator)

    def __iter__(self):
        return self

    def __load_dataset(self):
        for bundle in self.bundle_generator:
            valid_boxes = BoxOperations.get_valid_boxes(bundle.boxes, bundle.image.shape)
            for i, box in enumerate(valid_boxes):
                #Remove box if box is too small
                if (box[2] - box[0] < self.min_original_rectangle[0]) or (box[3] - box[1] < self.min_original_rectangle[1]):
                    continue

                loosened_box = BoxOperations.loosen_box(box, self.reference_rectangle, bundle.image.shape)
                if not loosened_box:
                    continue

                optimized_box = loosened_box if (self.mean is None or self.inv_cov is None) else BoxOperations.optimize_box(
                    box=loosened_box, mean=self.mean, inv_cov=self.inv_cov, depth=bundle.depth,
                    instances=bundle.instances, reference_rectangle=self.reference_rectangle,
                    constraint_box=box, depth_normalization_func=self.depth_normalization_func,
                    base_image_shape=bundle.image.shape, direction=self.optimize_direction
                )

                image = resize(bundle.image[optimized_box[0]:optimized_box[2], optimized_box[1]:optimized_box[3], :], (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST)

                depth = resize(
                    bundle.depth[optimized_box[0]:optimized_box[2], optimized_box[1]:optimized_box[3]], 
                    (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                instance = resize(
                    bundle.instances[optimized_box[0]:optimized_box[2], optimized_box[1]:optimized_box[3]], 
                    (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST
                )

                camera = array([
                    [self.reference_rectangle[1]/(optimized_box[3]-optimized_box[1]), 0, 0],
                    [0, self.reference_rectangle[0]/(optimized_box[2]-optimized_box[0]), 0],
                    [0, 0, 1]
                ]) @ (bundle.camera + block([zeros((3,2)), array([-optimized_box[1], -optimized_box[0], 0]).reshape((-1,1))]))

                yield Car(image, depth, instance, camera, bundle.camera, i, self.depth_normalization_func)
