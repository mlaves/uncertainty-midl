# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2019

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from glob import glob
from skimage import io, color
from torchvision import transforms


class KermanyDataset(Dataset):
    """
    Loads the Kermany OCT data set
    """

    def __init__(self, img_dir, crop_to=(496, 496), resize_to=(512, 512), file_ext='.jpeg', color=False):
        """
        Given the root directory of the dataset, this function initializes the
        data set

        :param img_dir: List with paths of raw images
        """

        self._crop_to = crop_to
        self._resize_to = resize_to
        self._color = color
        self._img_dir = img_dir
        self._class_list = glob(self._img_dir + "/*")
        self._class_list = sorted([c.split('/')[-1] for c in self._class_list])  # crop full paths
        self._img_file_names = []
        for c in self._class_list:
            self._img_file_names += sorted(glob(self._img_dir + f'/{c}/*' + file_ext))

    def __len__(self):
        return len(self._img_file_names)

    def __getitem__(self, idx):
        x = io.imread(self._img_file_names[idx])
        if self._color:
            x = color.gray2rgb(x)
        y = torch.zeros(1).long()  # CrossEntropyLoss does not expect a one-hot encoded vector, but class indices

        x = np.atleast_3d(x)  # I love numpy!

        for i, c in enumerate(self._class_list):
            if c in self._img_file_names[idx]:
                y = i

        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(self._crop_to),
            transforms.Resize(self._resize_to),
            transforms.ToTensor()
        ])

        x = trans(x)

        return x, y


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dataset = KermanyDataset("./../../data/test/")
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1)
    for i_batch, (data, target) in enumerate(data_loader):
        print(i_batch, data.size(), data.type(),
              target.size(), target.type())
        plt.imshow(data.data.cpu().numpy()[0, 0])
        print(target.data.cpu())
        plt.pause(0.25)
        plt.clf()
