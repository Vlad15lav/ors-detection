import argparse
import matplotlib.pyplot as plt
import yaml

import torch

from PIL import Image
from torchvision.transforms import Compose

from data.augment import Normalizer, Resizer
from model.yolo import YoloV3
from tools.inference import model_inference, draw_boxes


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser("Optical Remote Sensing Detection - " /
                                     "Vlad15lav")
    parser.add_argument("-p", "--path", type=str, help="path image")
    parser.add_argument("--cfg", type=str, default="dior", help="name cfg")
    parser.add_argument("--img_size", type=int, default=512, help="image size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_args()
    cfg = Params(f"projects/{opt.cfg}.yml")

    # Init model
    model = YoloV3(len(cfg.mask), cfg.anchors, opt.img_size)

    # Load weights
    try:
        model.load_state_dict(torch.load(f"states/{opt.cfg}_weights.pth"))
    except FileNotFoundError:
        print(
            "Weights is not found. You should move the weights to \
            /states/{name_proj}_weights.pth"
        )

    transform_test = Compose([Normalizer(cfg.mean, cfg.std),
                              Resizer(opt.img_size)])

    # Load image
    img = Image.open(opt.path)

    # Get predict
    bbox, cls_label, text_statistic = model_inference(model, img)
    img_draw = draw_boxes(img, bbox, cls_label)
    plt.imshow(img_draw)
    plt.show()
