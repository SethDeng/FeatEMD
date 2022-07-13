import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ...IntelligentCrop import IntelligentCrop
import os
import numpy as np
from Models.utils import *

class tieredImageNet(Dataset):

    def __init__(self, setname, args=None):
        TRAIN_PATH = osp.join(args.data_dir, 'tiered_imagenet/train')
        VAL_PATH = osp.join(args.data_dir, 'tiered_imagenet/val')
        TEST_PATH = osp.join(args.data_dir, 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Unkown setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        image_size = 84

        # EMD_cc setting
        self.generate_box = False
        self.num_patch = args.num_patch
        self.alpha = args.alpha
        self.outer_num = args.outer_num
        self.all_in_num = args.all_in_num

        self.boxes_outer = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)

        # EMD sampling set
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size = image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])

        # IntelligentCrop sampling set(8+1?)
        self.transform_cc = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])
        self.transform_all_in = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):  # return the ith data in the set.
        path, label = self.data[i], self.label[i]
        
        if self.generate_box:
            image = self.transform_all_in(Image.open(path).convert('RGB'))
            return image, label
        else:
            box_outer = self.boxes_outer[i].float().tolist()
            patch_list = []
            for j in range(self.num_patch):
                if j < self.outer_num:
                    img = IntelligentCrop(alpha=self.alpha, size=84, scale=(0.2, 1.0))(Image.open(path).convert('RGB'), box_outer)
                    patch_list.append(self.transform_cc(img))
                elif j >= self.outer_num and j < (self.num_patch - self.all_in_num):
                    patch_list.append(self.transform(Image.open(path).convert('RGB')))
                else:
                    patch_list.append(self.transform_all_in(Image.open(path).convert('RGB')))
            patch_list=torch.stack(patch_list,dim=0)
            return patch_list, label


if __name__ == '__main__':
    pass
