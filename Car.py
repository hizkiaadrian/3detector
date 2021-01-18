class Car:
    def __init__(self, base_image_path, image, depth, box, instance, camera, car_index):
        self.base_image_path = base_image_path
        self.image = image
        self.depth = depth
        self.box = box
        self.instance = instance
        self.camera = camera
        self.car_index = car_index