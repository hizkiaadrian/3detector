from BundleGenerator import BundleGenerator
from Car import Car
from os import listdir
from cv2 import resize, INTER_NEAREST
from numpy import array, zeros
from numpy.linalg import inv

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

class Dataset:
    def __init__(self, base_dir, date = None, reference_rectangle = (64, 128), min_original_rectangle = (16, 32)):
        if date is not None and date not in listdir(base_dir):
            raise ValueError
        self.bundle_generator = BundleGenerator(base_dir).load(date)
        self.reference_rectangle = reference_rectangle
        self.min_original_rectangle = min_original_rectangle

    def load_dataset(self):
        for bundle in self.bundle_generator:
            valid_boxes = get_valid_boxes(bundle.boxes, bundle.image.shape)
            for i, box in enumerate(valid_boxes):
                #Remove box if box is too small
                if (box[2] - box[0] < self.min_original_rectangle[0]) or (box[3] - box[1] < self.min_original_rectangle[1]):
                    continue

                #Resize box to have the correct h/w ratio
                hw_ratio = self.reference_rectangle[0] / self.reference_rectangle[1]
                box_ratio = (box[2] - box[0]) / (box[3] - box[1])

                #Either resize height or width depending on the original dimension
                if box_ratio > hw_ratio:
                    additional_width = ((box[2] - box[0]) / hw_ratio - (box[3] - box[1])) / 2
                    #Remove box if resized box would be truncated
                    if (box[1] - additional_width < 0) or (box[3] + additional_width >= bundle.image.shape[1]):
                        continue
                    
                    box[1] -= int(additional_width)
                    box[3] += int(additional_width)
                elif box_ratio < hw_ratio:
                    additional_height = ((box[3] - box[1]) * hw_ratio - (box[2] - box[0])) / 2
                    #Remove box if resized box would be truncated
                    if (box[0] - additional_height < 0) or (box[2] + additional_width >= bundle.image.shape[0]):
                        continue
                    
                    box[0] -= int(additional_height)
                    box[2] += int(additional_height)

                #Retrieve true/world coordinates
                camera_inv = inv(bundle.camera)

                world_coordinates = zeros((box[2]-box[0],box[3]-box[1],3))
                for x in range(box[1], box[3]):
                    for y in range(box[0], box[2]):
                        try:
                            world_coordinates[y-box[0], x-box[1],:] = camera_inv @ [x, y, 1] * bundle.depth[y, x]
                        except:
                            continue

                image = bundle.image[box[0]:box[2], box[1]:box[3], :]
                depth = bundle.depth[box[0]:box[2], box[1]:box[3]]
                instance = bundle.instances[box[0]:box[2], box[1]:box[3]]

                image = resize(image, (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST)
                depth = resize(depth, (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST)
                instance = resize(instance, (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST)
                world_coordinates = resize(world_coordinates, (self.reference_rectangle[1], self.reference_rectangle[0]), interpolation=INTER_NEAREST)

                yield Car(bundle.img_path, image, depth, instance, world_coordinates, i)
