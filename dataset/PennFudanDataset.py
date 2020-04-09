# dataset url = https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
import os
import requests
import zipfile
import numpy as np
import torch
from PIL import Image


def download_dataset(url, save_path, chunk_size=512):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def extract_dataset(zipname):
    with zipfile.ZipFile(zipname) as zf:
        zf.extractall()


class PennFudanDataset(object):
    """
    Penn Fundan Dataset Loader for Pytorch
    """

    def __init__(self, rootdir=None, transforms=None, debug=False):
        """
        Initiatize the dataset loader. Stores all data paths as string.
        :param rootdir: main, top directory of dataset
        :param transforms: augmentations
        """
        current_dir = os.getcwd()
        dataset_name = 'PennFudanPed'
        if rootdir is None:
            # I providing dataset inside of project.
            rootdir = current_dir + dataset_name
            pass

        url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
        zipname = current_dir + "\\" + dataset_name + ".zip"
        download_dataset(url, zipname)
        extract_dataset(zipname)
        if os.path.isdir(rootdir) == False:
            # if folder is empty download and extract it.
            download_dataset(url, zipname)
            extract_dataset(zipname)

        self.images_rootdir = rootdir + "PNGImages"
        self.masks_rootdir = rootdir + "PedMasks"
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(rootdir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(rootdir, "PedMasks"))))

        if debug:
            print(current_dir)
            print(rootdir)
            print(self.images)
            print(self.masks)
        assert len(self.images) == len(
            self.masks)  # Training might be broken, if number of images and number of masks not equal

    def __getitem__(self, index):
        self.image = Image.open(self.images_rootdir + self.images[index])
        self.mask = Image.open(self.masks_rootdir + self.masks[index])
        pair = (self.image, self.mask)
        return pair  # change it to better representation

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return "PennFudanDataset"


if __name__ == "__main__":
    rootdir = 'C://Users//selcu//PycharmProjects//SingleShotMultiboxDetector_demo//dataset//PennFudanPed//'
    transforms = None

    dataset = PennFudanDataset(rootdir, transforms)
    print(dataset.images)
    print(dataset.masks)
    print(dataset[0])
