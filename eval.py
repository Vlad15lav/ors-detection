import argparse
import os
import yaml
import math
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from model.yolo import YoloV3
from model.weight import *
from data.dataloader import Dataset
from data.augment import Normalizer, Resizer
from tools.metrics import evaluate

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Optical Remote Sensing Detection - Vlad15lav')
    parser.add_argument('-p', '--path', type=str, help='path dataset')
    parser.add_argument('--cfg', type=str, help='name cfg')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('-bs', '--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--n_work', type=int, default=8, help='number of gpu')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    cfg = Params(f'projects/{opt.cfg}.yml')

    model = YoloV3(len(cfg.mask), cfg.anchors, opt.img_size)
    model.apply(weights_init_normal)
    model = model.cuda()

    # load weights
    try:
        model.load_state_dict(torch.load(f'states/{opt.cfg}_weights.pth'))
    except FileNotFoundError:
        print('Weights is not found. You should move the weights to \
            /states/{name_proj}_weights.pth')

    transform_test=Compose([Normalizer(cfg.mean, cfg.std),
        Resizer(opt.img_size)])

    testset = Dataset(f'{opt.path}/test.txt', opt, cfg, transform=transform_test)
    test_num = len(testset)
    testval_indicies = np.arange(test_num)
     
    # split train, validation, test
    validation_sampler = SubsetRandomSampler(testval_indicies[:int(test_num * 0.5)])
    test_sampler = SubsetRandomSampler(testval_indicies[int(test_num * 0.5):])

    TestLoader = DataLoader(testset, batch_size=opt.batch_size, collate_fn=testset.collate_fn,
        sampler=test_sampler, num_workers=opt.n_work)

    precision, recall, AP, f1, ap_class = evaluate(
                model,
                TestLoader,
                iou_thres=cfg.iou_thres,
                conf_thres=cfg.conf_thres,
                nms_thres=cfg.nms_thres,
                img_size=opt.img_size
            )

    print('Eval mAP - ' + str(AP.mean() * 100))
    print('AP by classes: {}'.format(AP * 100))