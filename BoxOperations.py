from cv2 import resize, INTER_NEAREST
from math import ceil
from enum import Enum

class Direction(Enum):
    VERTICAL = 0
    HORIZONTAL = 1
    ALL = 2

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