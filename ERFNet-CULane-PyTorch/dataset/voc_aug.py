import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VOCAugDataSet(Dataset):
    def __init__(self, dataset_path='/home/ywang/dataset/CULane/list', data_list='train', transform=None):

        self.is_testing = data_list == 'test_img' # 'val'
        with open(os.path.join(dataset_path, data_list + '.txt')) as f:
            self.img_list = []
            self.img = []
            self.label_list = []
            self.exist_list = []
            for line in f:
                self.img.append(line.strip().split(" ")[0])
                self.img_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[0])
                if not self.is_testing:
                    self.label_list.append(dataset_path.replace('/list', '') + line.strip().split(" ")[1])
                    self.exist_list.append(np.array([int(line.strip().split(" ")[2]), int(line.strip().split(" ")[3]), int(line.strip().split(" ")[4]), int(line.strip().split(" ")[5])]))

        self.img_path = dataset_path
        self.gt_path = dataset_path
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        #print (os.path.join(self.img_path, self.img_list[idx]))
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx])).astype(np.float32)
        if not self.is_testing:
            label = cv2.imread(os.path.join(self.gt_path, self.label_list[idx]), cv2.IMREAD_UNCHANGED)
            exist = self.exist_list[idx]
            label = label[240:, :]
            label = label.squeeze()
        image = image[240:, :, :]
        if self.transform:
            if self.is_testing:
                [image] = self.transform([image])
            else:
                image,label = self.transform(image,label)
                label = torch.from_numpy(label).contiguous().long()
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        if self.is_testing:
            return image , self.img[idx]
        else:
            return image, label, exist
