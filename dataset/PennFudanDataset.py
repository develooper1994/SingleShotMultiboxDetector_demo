# dataset url = https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
import os
import requests
import zipfile
import numpy as np
import torch
from PIL import Image


def download_dataset(url, save_path, chunk_size=512):
    """
    Downloads Penn Fudan Dataset.
    :param url: url of Penn Fudan Dataset
    :param save_path: ful path + filename + file extention
    :param chunk_size: it is required if file partition is too small (embedded systems)
    :return: None
    """
    # TODO: extract header info to get download size
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def extract_dataset(zipname):
    """
    Extract the Penn Fudan Dataset.
    :param zipname: ful path + filename + file extention
    :return: None
    """
    with zipfile.ZipFile(zipname) as zf:
        zf.extractall()


class PennFudanDataset(object):
    """
    Penn Fundan Dataset Loader for Pytorch
    """

    def __init__(self, rootdir=None, transforms=None, debug=False):
        """
        Initiatize the dataset loader. Stores all data paths as string.
        Downloads and extracts if PennFudanDataset.zip not exist.
        throws assertation if file lenghts are not equal
        :param rootdir: main, top directory of dataset
        :param transforms: augmentations
        """
        current_dir = os.getcwd()
        dataset_name = 'PennFudanPed'
        if rootdir is None:
            # I providing dataset inside of project.
            rootdir = current_dir + dataset_name  # load directory

        url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
        zipname = current_dir + "\\" + dataset_name + ".zip"
        if os.path.isdir(rootdir) == False:
            # if folder is empty download and extract it.
            download_dataset(url, zipname)
            extract_dataset(zipname)

        self.images_rootdir = rootdir + "PNGImages//"
        self.masks_rootdir = rootdir + "PedMasks//"
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(rootdir, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(rootdir, "PedMasks"))))

        if debug:
            print(current_dir)
            print(rootdir)
            print(zipname)
            print(self.images)
            print(self.masks)
        assert len(self.images) == len(
            self.masks)  # Training might be broken, if number of images and number of masks not equal

    def __getitem__(self, index):
        """
        get item pairs. (image, mask)
        # not: This function is not a good for production. It is fast prototyping.
        :param index: index of data
        :return: None
        """
        # TODO: use pescador to faster multiprocess data streaming.
        # TODO: First read all dataset, load before __getitem__ and encode to one-hot-tensor
        image = Image.open(self.images_rootdir + self.images[index]).convert("RGB")  # get image of index number
        mask = Image.open(self.masks_rootdir + self.masks[index])  # get mask image of index number
        mask = np.array(mask)  # convert to numpy array.
        obj_ids = np.unique(mask)  # Eliminate colors and extract object ids
        obj_ids = obj_ids[1:]  # Eliminate background. No needed.
        masks = mask == obj_ids[:, None, None]  # split the color-encoded mask of binary image, get object ids

        # get bounding boxes
        boxes = self.bounding_box(masks, obj_ids)

        # convert everything into a torch.Tensor
        boxes, labels, masks, image_id = self.convert_to_torch(boxes, index, masks, obj_ids)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        # information about target
        target = {"boxes": boxes,
                  "labels": labels,
                  "masks": masks,
                  "image_id": image_id,
                  "area": area,
                  "iscrowd": iscrowd}

        # apply transform if not None
        if self.transforms is not None:
            # TODO: Write transform function to wrap all of them
            image, target = self.transforms(image, target)

        return image, target  # return pair

    def convert_to_torch(self, boxes, index, masks, obj_ids):
        """
        Convert everything into a torch.Tensor
        :param boxes: bounding boxes as numpy array
        :param index: index of object
        :param masks: numpy mask array
        :param obj_ids: object ids
        :return: bounding boxes,
        """
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        return boxes, labels, masks, image_id

    def bounding_box(self, masks, obj_ids):
        """
        Get bounding box min and max coordinate to construct the BOX for each object
        :param masks: split the color-encoded mask of binary image, get object ids
        :param obj_ids: object ids
        :return: bounding box
        """
        boxes = []
        for i in range(len(obj_ids)):
            position = np.where(masks[i])
            # x
            x_min = np.min(position[1])
            x_max = np.max(position[1])
            # y
            y_min = np.min(position[0])
            y_max = np.max(position[0])
            boxes.append([x_min, y_min, x_max, y_max])
        return boxes

    def __len__(self):
        assert len(self.images) == len(self.masks)
        return len(self.imgs)

    def __repr__(self):
        return "PennFudanDataset"


if __name__ == "__main__":
    rootdir = 'C://Users//selcu//PycharmProjects//SingleShotMultiboxDetector_demo//dataset//PennFudanPed//'
    transforms = None

    dataset = PennFudanDataset(rootdir, transforms)
    # print(dataset.images)
    # print(dataset.masks)
    print(dataset[0])
