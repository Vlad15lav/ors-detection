import argparse
import os
import pickle
import torch
import yaml
import math
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
from IPython import display

from model.yolo import YoloV3
from model.weight import weights_init_normal
from data.dataloader import Dataset
from data.augment import Normalizer, Flip, RandomHSV, Resizer, Compose
from tools.metrics import evaluate


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser("Optical Remote Sensing Detection - " /
                                     "Vlad15lav")
    parser.add_argument("-p", "--path", type=str, help="path dataset")
    parser.add_argument("--cfg", type=str, help="name cfg")
    parser.add_argument("--img_size", type=int, default=512, help="image size")
    parser.add_argument("--epoches", type=int, default=50,
                        help="number of epoches")
    parser.add_argument("--adam", help="Adam optimizer", action="store_true")
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=12, help="batch size")
    parser.add_argument("--n_work", type=int, default=2,
                        help="number of gpu")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=0.0005,
                        help="weight decay")
    parser.add_argument("--load_train", help="continue training",
                        action="store_true")
    parser.add_argument("--debug", help="debug training", action="store_true")

    args = parser.parse_args()
    return args


def train(train_loader, val_loader, model,
          optimizer, opt, cfg, scheduler=None):
    nb = len(train_loader)
    n_burn = max(3 * nb, 500)

    accumulate = max(round(cfg.accumulate_batch / opt.batch_size), 1)

    train_loss, val_precision, val_recall, val_mAP, val_f1 = [], [], [], [], []

    if not os.path.exists("states"):
        os.makedirs("states")

    if opt.load_train:
        f_log = open(f"states/{opt.cfg}_log.pickle", "rb")
        obj = pickle.load(f_log)
        train_loss, val_precision, val_recall, val_mAP, val_f1 = obj
        f_log.close()

    for epoch in tqdm(range(0, opt.epoches)):
        display.clear_output(wait=True)
        # train
        model.train(True)
        loss_batch = []
        for i, (_, imgs, targets) in enumerate(train_loader):
            ni = i + nb * epoch
            imgs = Variable(imgs.cuda())
            imgs = imgs.float() / 255.0
            targets = Variable(targets.cuda(), requires_grad=False)

            # warmup
            if ni <= n_burn:
                xi = [0, n_burn]
                model.gr = np.interp(ni, xi, [0.0, 1.0])
                accumulate = max(
                    1,
                    np.interp(
                        ni, xi, [1, cfg.accumulate_batch / opt.batch_size]
                    ).round(),
                )
                for j, p_model in enumerate(optimizer.param_groups):
                    p_model["lr"] = np.interp(
                        ni,
                        xi,
                        [0.1 if j == 2 else 0.0,
                         p_model["initial_lr"] * lf(epoch)],
                    )
                    p_model["weight_decay"] = np.interp(
                        ni, xi, [0.0, opt.lr if j == 1 else 0.0]
                    )
                    if "momentum" in p_model:
                        p_model["momentum"] = np.interp(ni, xi,
                                                        [0.9, cfg.momentum])

            # forward
            loss, outputs = model(imgs, targets)

            # backward with scale loss
            loss *= opt.batch_size / cfg.loss_scale
            loss.backward()

            loss_batch.append(loss.item())

            # accumulate grad
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        train_loss.append(np.mean(loss_batch))

        # valid and waiting warmup
        if epoch % cfg.valid_time == 0 or epoch > 2:
            model.train(False)
            with torch.no_grad():
                precision, recall, AP, f1, ap_class = evaluate(
                    model,
                    val_loader,
                    iou_thres=cfg.iou_thres,
                    conf_thres=cfg.conf_thres,
                    nms_thres=cfg.nms_thres,
                    img_size=opt.img_size,
                )

                val_precision.append(precision.mean())
                val_recall.append(recall.mean())
                val_mAP.append(AP.mean())
                val_f1.append(f1.mean())

            if opt.debug:
                print(
                    f"Epoche {epoch}: Train loss {train_loss[-1]},\
                    Val Precision {val_precision[-1]}, Val Recall \
                    {val_recall[-1]}, Val mAP {val_mAP[-1]}, \
                    Val F1 {val_f1[-1]}"
                )
        elif opt.debug:
            print(f"Epoche {epoch}: Train loss {train_loss[-1]}")

        # save optimizer
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            f"states/checkponit_{opt.cfg}.pth",
        )

        # save best weights
        if len(val_mAP) > 1 and val_mAP[-1] > np.max(val_mAP[:-1]):
            torch.save(model.state_dict(), f"states/{opt.cfg}_weights.pth")

        # log history
        lists = (train_loss, val_precision, val_recall, val_mAP, val_f1)
        f_log = open(f"states/{opt.cfg}_log.pickle", "wb")
        pickle.dump(lists, f_log)
        f_log.close()

        # ploting
        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].set_title("Loss")
        axes[0].plot(train_loss, label="Train loss")
        axes[0].legend()
        axes[1].set_title("Val metrics")
        axes[1].plot(val_precision, label="Precision")
        axes[1].plot(val_recall, label="Recall")
        axes[1].plot(val_mAP, label="mAP")
        axes[1].plot(val_f1, label="F1 score")
        axes[1].legend()
        plt.show()

    print("\nFinal mAP: ", val_mAP[-1])


if __name__ == "__main__":
    opt = get_args()
    cfg = Params(f"projects/{opt.cfg}.yml")

    model = YoloV3(len(cfg.mask), cfg.anchors, opt.img_size)
    model.apply(weights_init_normal)
    model = model.cuda()

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.lr,
                                     weight_decay=opt.wd)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, momentum=cfg.momentum, nesterov=True
        )

    # scheduler CosLR
    lf = (
        lambda x: (((1 + math.cos(x * math.pi / opt.epoches)) / 2) ** 1.0) *
        0.95 + 0.05
    )
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = -1

    # continue training
    if opt.load_train:
        model.load_state_dict(torch.load(f"states/{opt.cfg}_weights.pth"))
        checkpoint = torch.load(f"states/checkponit_{opt.cfg}.pth")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    transform_train = Compose(
        [
            RandomHSV(cfg.hsvlow, cfg.hsvhigh),
            Flip(cfg.flip_x, cfg.flip_y),
            Normalizer(cfg.mean, cfg.std),
            Resizer(opt.img_size),
        ]
    )
    transform_test = Compose([Normalizer(cfg.mean, cfg.std),
                              Resizer(opt.img_size)])

    trainset = Dataset(f"{opt.path}/train.txt", opt, cfg,
                       transform=transform_train)
    testset = Dataset(f"{opt.path}/test.txt", opt, cfg,
                      transform=transform_test)

    train_num = len(trainset)
    test_num = len(testset)

    train_indicies = np.arange(train_num)
    testval_indicies = np.arange(test_num)
    np.random.shuffle(train_indicies)

    # split train, validation, test
    train_sampler = SubsetRandomSampler(train_indicies)
    validation_sampler = SubsetRandomSampler(
        testval_indicies[: int(test_num * 0.5)]
        )
    test_sampler = SubsetRandomSampler(
        testval_indicies[int(test_num * 0.5):]
        )

    TrainLoader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        collate_fn=trainset.collate_fn,
        sampler=train_sampler,
        num_workers=opt.n_work,
    )
    ValidationLoader = DataLoader(
        testset,
        batch_size=opt.batch_size,
        collate_fn=trainset.collate_fn,
        sampler=validation_sampler,
        num_workers=opt.n_work,
    )

    train(TrainLoader, ValidationLoader, model, optimizer, opt, cfg, scheduler)
