import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, rootdir, transforms):
        self.images = list(sorted(os.listdir(os.path.join(rootdir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(rootdir, "PedMasks"))))

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return "PennFudanDataset"


if __name__ == "__main__":
    pass
