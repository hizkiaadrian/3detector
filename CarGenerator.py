from BundleGenerator import BundleGenerator
from Car import Car
from os import listdir
from cv2 import resize, INTER_NEAREST
from numpy import array, zeros, block, median
from numpy.linalg import inv
from scipy.stats import mode
from math import ceil
from enum import Enum

def is_occluded(larger_box, smaller_box):
    return not (larger_box[2] <= smaller_box[0] or larger_box[0] >= smaller_box[2] or larger_box[3] <= smaller_box[1] or larger_box[1] >= smaller_box[3])

def get_valid_boxes(boxes, img_dimension):
    boxes = sorted(boxes, key=lambda x: x[0] - x[2])

    height = img_dimension[0]
    width = img_dimension[1]

    #Remove occluded boxes
    box_props = sorted([{'box':box, 'occ':False} for box in boxes], key=lambda x: x['box'][0] - x['box'][2])
    for i, box in enumerate(boxes):
        for j in range(i+1, len(box_props)):
            if is_occluded(box, box_props[j]['box']):
                box_props[j]['occ'] = True

    #Remove truncated boxes
    return list(
        filter(lambda box: box[0] >= 0 and box[1] >= 0 and box[2] < height and box[3] < width, 
                [box['box'] for box in box_props if not box['occ']]
            )
        )

def loosen_box(box, reference_rectangle, base_image_shape):
    hw_ratio = reference_rectangle[0] / reference_rectangle[1]
    box_ratio = (box[2] - box[0]) / (box[3] - box[1])

    new_box = box.copy()

    #Either resize height or width depending on the original dimension
    if box_ratio > hw_ratio:
        additional_width = ((box[2] - box[0]) / hw_ratio - (box[3] - box[1])) / 2
        #Remove box if resized box would be truncated
        if (box[1] - additional_width < 0) or (box[3] + additional_width >= base_image_shape[1]):
            return None
        
        new_box[1] -= int(additional_width)
        new_box[3] += int(additional_width)
    elif box_ratio < hw_ratio:
        additional_height = ((box[3] - box[1]) * hw_ratio - (box[2] - box[0])) / 2
        #Remove box if resized box would be truncated
        if (box[0] - additional_height < 0) or (box[2] + additional_height >= base_image_shape[0]):
            return None
        
        new_box[0] -= int(additional_height)
        new_box[2] += int(additional_height)

    return new_box

def get_iou(box1, box2):
    intersection = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    union = (box1[2]-box1[0]) * (box1[3]-box1[1]) + (box2[2]-box2[0]) * (box2[3]-box2[1]) - intersection
    return intersection / union

class Direction(Enum):
    VERTICAL = 0
    HORIZONTAL = 1
    ALL = 2

def optimize_box(box,
                 mean,
                 inv_cov,
                 depth, 
                 instances, 
                 reference_rectangle, 
                 constraint_box, 
                 base_image_shape,
                 depth_normalization_func, 
                 direction=Direction.VERTICAL, 
                 min_intersection_ratio=0.8):
    crop = resize(
                    depth[box[0]:box[2], box[1]:box[3]], 
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
    instance = resize(
                    instances[box[0]:box[2], box[1]:box[3]],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )

    normalized_crop = depth_normalization_func(crop, instance)
    min_score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)

    optimal_box = box.copy()

    if direction == Direction.VERTICAL:
        for top_y in range(box[0], constraint_box[0]):
            bottom_y = box[2] + top_y - box[0]
            if bottom_y < base_image_shape[0] and get_iou(box, [top_y, box[1], bottom_y, box[3]]) > min_intersection_ratio:
                crop = resize(
                    depth[top_y:bottom_y, box[1]:box[3]],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                instance = resize(
                    instances[top_y:bottom_y, box[1]:box[3]],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                normalized_crop = depth_normalization_func(crop, instance)
                score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)
                if score < min_score:
                    min_score = score
                    optimal_box = [top_y, box[1], bottom_y, box[3]]
            else:
                break
        for top_y in range(box[0], -1, -1):
            bottom_y = box[2] + top_y - box[0]
            if bottom_y > constraint_box[2] and get_iou(box, [top_y, box[1], bottom_y, box[3]]) > min_intersection_ratio:
                crop = resize(
                    depth[top_y:bottom_y, box[1]:box[3]],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                instance = resize(
                    instances[top_y:bottom_y, box[1]:box[3]],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                normalized_crop = depth_normalization_func(crop, instance)
                score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)
                if score < min_score:
                    min_score = score
                    optimal_box = [top_y, box[1], bottom_y, box[3]]
            else:
                break

    elif direction == Direction.HORIZONTAL:
        for top_x in range(box[1], constraint_box[1]):
            bottom_x = box[3] + top_x - box[1]
            if bottom_x < base_image_shape[1] and get_iou(box, [box[0], top_x, box[2], bottom_x]) > min_intersection_ratio:
                crop = resize(
                    depth[box[0]:box[2], top_x:bottom_x],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                instance = resize(
                    instances[box[0]:box[2], top_x:bottom_x],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                normalized_crop = depth_normalization_func(crop, instance)
                score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)
                if score < min_score:
                    min_score = score
                    optimal_box = [box[0], top_x, box[2], bottom_x]
            else:
                break
        for top_x in range(box[1], -1, -1):
            bottom_x = box[3] + top_x - box[1]
            if bottom_x > constraint_box[3] and get_iou(box, [box[0], top_x, box[2], bottom_x]) > min_intersection_ratio:
                crop = resize(
                    depth[box[0]:box[2], top_x:bottom_x],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                instance = resize(
                    instances[box[0]:box[2], top_x:bottom_x],
                    (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                )
                normalized_crop = depth_normalization_func(crop, instance)
                score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)
                if score < min_score:
                    min_score = score
                    optimal_box = [box[0], top_x, box[2], bottom_x]
            else:
                break

    elif direction == Direction.ALL:
        hw_ratio = reference_rectangle[0] / reference_rectangle[1]
        for top_y in range(constraint_box[0]):
            for top_x in range(constraint_box[1]):
                for bottom_y in range(constraint_box[2], base_image_shape[0]):
                    bottom_x = top_x + (bottom_y - top_y) / hw_ratio
                    if not bottom_x.is_integer():
                        continue
                    elif bottom_x < base_image_shape[1] and get_iou(box, [top_y, top_x, bottom_y, ceil(bottom_x)]) > min_intersection_ratio:
                        crop = resize(
                            depth[top_y:bottom_y, top_x:ceil(bottom_x)],
                            (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                        )
                        instance = resize(
                            instances[top_y:bottom_y, top_x:ceil(bottom_x)],
                            (reference_rectangle[1], reference_rectangle[0]), interpolation=INTER_NEAREST
                        )
                        normalized_crop = depth_normalization_func(crop, instance)
                        score = (normalized_crop.ravel() - mean).T @ inv_cov @ (normalized_crop.ravel() - mean)
                        if score < min_score:
                            min_score = score
                            optimal_box = [top_y, top_x, bottom_y, ceil(bottom_x)]
                    else:
                        break

    return optimal_box

class CarGenerator:
    def __init__(self, 
                 base_dir, 
                 date = None, 
                 reference_rectangle = (64, 128), 
                 min_original_rectangle = (16, 32), 
                 depth_normalization_func=None, 
                 mean = None, 
                 cov = None, 
                 optimize_direction = Direction.ALL
                ):
        if date is not None and date not in listdir(base_dir):
            raise ValueError
        self._bg_object = BundleGenerator(base_dir)
        self.bundle_generator = self._bg_object.load(date)
        self.reference_rectangle = reference_rectangle
        self.min_original_rectangle = min_original_rectangle
        self.depth_normalization_func = depth_normalization_func if depth_normalization_func is not None else lambda depth, instance : depth / median(depth)
        self.mean = mean
        self.cov = cov
        self.inv_cov = inv(cov) if cov is not None else None
        self.optimize_direction = optimize_direction

    def load_dataset(self):
        for bundle in self.bundle_generator:
            valid_boxes = get_valid_boxes(bundle.boxes, bundle.image.shape)
            for i, box in enumerate(valid_boxes):
                #Remove box if box is too small
                if (box[2] - box[0] < self.min_original_rectangle[0]) or (box[3] - box[1] < self.min_original_rectangle[1]):
                    continue

                loosened_box = loosen_box(box, self.reference_rectangle, bundle.image.shape)
                if not loosened_box:
                    continue

                optimized_box = loosened_box if (self.mean is None or self.inv_cov is None) else optimize_box(
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
