import numpy as np
import cv2
import random
import torch

from tools.box import xywh2xyxy, xyxy2xywh

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes):
        for t in self.transforms:
            image, boxes = t(image, boxes)
        return image, boxes

class Resizer(object):
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, img, boxes):
        height, width, _ = img.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        img = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = img

        boxes[:] *= scale

        return torch.from_numpy(new_image).to(torch.float32), torch.from_numpy(boxes)

class RandomHSV(object):
    def __init__(self, low=0.5, high=3.0):
        self.low = low
        self.high = high

    def __call__(self, img, boxes):
        value = random.uniform(self.low, self.high)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1] * value
        hsv[:,:,1][hsv[:,:,1] > 255] = 255
        hsv[:,:,2] = hsv[:,:,2] * value 
        hsv[:,:,2][hsv[:,:,2] > 255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img, boxes

class Flip(object):
    def __init__(self, flip_x=0.5, flip_y=0.5):
        self.flip_x = flip_x
        self.flip_y = flip_y

    def __call__(self, img, boxes):
        boxes = xyxy2xywh(boxes)
        if random.random() < self.flip_x:
            img = np.fliplr(img)
            boxes[:, 0] = img.shape[1] - boxes[:, 0]

        if random.random() < self.flip_y:
            img = np.flipud(img)
            boxes[:, 1] = img.shape[0] - boxes[:, 1]
        boxes = xywh2xyxy(boxes)
        return img, boxes

class Normalizer(object):
    def __init__(self, mean=[0.39, 0.40, 0.36], std=[0.155 , 0.144, 0.141]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, img, boxes):
        return (img.astype(np.float32) / 255.0 - self.mean) / self.std, boxes