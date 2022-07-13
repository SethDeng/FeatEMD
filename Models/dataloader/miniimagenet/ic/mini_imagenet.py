import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from ...IntelligentCrop import IntelligentCrop
import os
import numpy as np
from Models.utils import *

class MiniImageNet(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'miniimagenet/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'miniimagenet/split')
        
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        # data_save = []
        label = []
        self.setname = setname
        self.wnids = []
        lb = -1

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data #data path of all data
        self.label = label #label of all data
        self.num_class = len(set(label))
        image_size = 84

        # EMD_cc setting
        self.generate_box = False
        self.num_patch = args.num_patch
        self.alpha = args.alpha
        self.outer_num = args.outer_num
        self.all_in_num = args.all_in_num

        self.boxes_outer = torch.tensor([0., 0., 1., 1.]).repeat(self.__len__(), 1)


        # if 'num_patch' not in vars(args).keys():
        #     print ('do not assign num_patch parameter, set as default: 9')
        #     self.num_patch=9
        # else:
        #     self.num_patch=args.num_patch

        # EMD sampling set
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size = image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])

        # IntelligentCrop sampling set
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

    def __getitem__(self, i):
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