import pykitti
from os import listdir
from glob import glob
from DataBundle import DataBundle

class BundleGenerator:
    def __init__(self, base_dir):
        self.dates = sorted(listdir(base_dir))
        self.base_dir = base_dir

    def load(self,date=None):
        if date is None:
            for date in self.dates:
                return self._load_image_paths(date)

        elif date in self.dates:
            return self._load_image_paths(date)

        else:
            raise ValueError

    def _load_image_paths(self, date):
        drives = sorted(
            list(
                map(
                    lambda x: x.split('_')[-2],
                    filter(lambda x: not x.endswith('txt'), listdir(f'{self.base_dir}/{date}'))
                )
            )
        )
        calib = pykitti.raw(self.base_dir, date, drives[0]).calib
        
        img_paths = sorted(
            sum(
                map(
                    lambda x: glob(
                        f'{self.base_dir}/{date}/{date}_drive_{x}_sync/image_02/data/*.jpg'),
                        drives
                    ), []
            )
        )
        
        for img_path in img_paths:
            yield DataBundle(img_path, calib)