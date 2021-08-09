# Optical Remote Sensing Detection
Objecte Detection in The Optical Remote Sensing🌍🛰

Course work at Kazan Federal University

## Requirements
```
pip install -U -r requirements.txt
```

Check this repository - [Object Detection Toolkit](https://github.com/Vlad15lav/ObjectDetection-Toolkit)

## Training model YOLOv3
Start training:
```
python train.py --path data/DIOR-full --cfg dior --img_size 512 --lr 0.005 --epoches 50 --batch_size 12 --debug
```
Continue training:
```
python train.py --path data/DIOR-full --cfg dior --img_size 512 --load_train --epoches 50 --batch_size 12 --debug
```

## Eval model
```
python eval.py --path data/DIOR-full --cfg dior --img_size 512 --batch_size 12
```
## Dior Dataset
<img src="/images/diorset.png" alt="drawing" width="550"/>

Load YOLOv3 weights:
```
wget https://github.com/Vlad15lav/Computer-Graphics/releases/download/animation/ffmpeg.exe -O states/dior_weights.pth
```
<img src="/images/test.gif" alt="drawing" width="550"/>

## References
- [Detection in Optical Remote Sensing Dataset](https://arxiv.org/ftp/arxiv/papers/1909/1909.00133.pdf)
- [You Only Look Once V3](https://arxiv.org/pdf/1804.02767.pdf)
