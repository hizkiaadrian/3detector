from BundleGenerator import BundleGenerator
from os import listdir

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

    return list(filter(lambda box: box[0] >= 0 and box[1] >= 0 and box[2] < height and box[3] < width, [box['box'] for box in box_props if not box['occ']]))
    
class Dataset:
    def __init__(self, base_dir, date = None):
        if date is not None and date not in listdir(base_dir):
            raise ValueError
        self.bundle_generator = BundleGenerator(base_dir).load(date)

    def load_dataset(self):
        for bundle in self.bundle_generator:
            valid_boxes = get_valid_boxes(bundle.boxes, bundle.image.shape)

