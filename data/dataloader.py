import numpy as np
import torch
import cv2
import os

from tools.box import xywh2xyxy, xyxy2xywh

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, opt, cfg, transform=None):
        self.cfg = cfg
        with open(path, 'r') as f:
            self.img_files = [x.replace('/', os.sep) for x in f.read().splitlines()
            	if os.path.splitext(x)[-1].lower() in ['.jpg', '.jpeg']]
        self.label_files = [x.replace('images', 'labels').\
        	replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files]
        n = len(self.img_files)
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        orig_w, orig_h = cfg.img_size

        for i, file in enumerate(self.label_files):
            try:
                with open(file, 'r') as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                    label = l.copy()
                    l[:, 1] = orig_w * (label[:, 1] - label[:, 3] / 2)
                    l[:, 2] = orig_h * (label[:, 2] - label[:, 4] / 2)
                    l[:, 3] = orig_w * (label[:, 1] + label[:, 3] / 2)
                    l[:, 4] = orig_h * (label[:, 2] + label[:, 4] / 2)
                    l = np.round(l)
            except:
                continue
            # not empty targets
            if l.shape[0]:
                self.labels[i] = l

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = self.get_img(index)
        height, width, channels = img.shape
        
        boxes, label = self.load_annotations(index)
        
        if self.transform is not None:
            img, boxes = self.transform(img, boxes)

        boxes = xyxy2xywh(boxes)

        boxes[:, [1, 3]] /= img.shape[0]
        boxes[:, [0, 2]] /= img.shape[1]

        target = np.zeros((boxes.shape[0], 6))
        target[:, 2:] = boxes
        target[:, 1] = label
        
        return img, torch.from_numpy(target), self.img_files[index]

    def num_classes(self):
        return len(self.cfg.mask)

    def label_to_name(self, label):
        return self.cfg.mask[label]

    def load_annotations(self, index):
        boxes = np.array(self.labels[index][:, 1:], np.float32).reshape((-1, 4))
        label = np.array(self.labels[index][:, 0], np.int64).reshape((-1, ))
        # remove invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        label = label[keep]

        return boxes, label

    def get_img(self, index):
        image_file = self.img_files[index]
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    @staticmethod
    def collate_fn(batch):
        img, target, path = zip(*batch)
        img = torch.stack(img, 0).permute(0, 3, 1, 2)
        for i, l in enumerate(target):
            l[:, 0] = i
        target = torch.cat(target, 0)
        return path, img, target.type(torch.FloatTensor)