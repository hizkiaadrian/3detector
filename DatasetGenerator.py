from BundleGenerator import BundleGenerator
from os import listdir

class Dataset:
    def __init__(self, base_dir, date = None):
        if date is not None and date not in listdir(base_dir):
            raise ValueError
        self.bundle_generator = BundleGenerator(base_dir).load(date)

    def load_dataset(self):
        pass