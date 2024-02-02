import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from tools.nms import non_max_suppression
from tools.box import rescale_boxes


MASK = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
    "overpass",
]
MASK_RUS = [
    "Самолет",
    "Аэропорт",
    "Бейсбольная площадка",
    "Баскетбольная площадка",
    "Мост",
    "Завод",
    "Дамба",
    "Зона обслуживания авто",
    "Пропускной пунтк",
    "Площадка для гольфа",
    "Поле для бега",
    "Гавань",
    "Корабль",
    "Стадион",
    "Резервуары хранения",
    "Теннисный корт",
    "ЖД станция",
    "Машина",
    "Ветрогенератор",
    "Эстакада",
]


def model_inference(model, image_original, conf_tresh: float = 0.35):
    # Resize фото под вход модели
    image = image_original.copy()
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    image = transforms.ToTensor()(image.convert("RGB"))
    if len(image.shape) != 3:
        image = image.unsqueeze(0)
        image = image.expand((3, image.shape[1:]))

    # Предсказание модели и NMS для удаление перекрывающих боксов
    predict = model(image[None, :])
    detections = non_max_suppression(predict,
                                     conf_thres=conf_tresh,
                                     nms_thres=0.5)

    max_orig_size = max(image_original.size)
    img_orig = np.array(
        image_original.resize(
            (max_orig_size, max_orig_size), Image.Resampling.LANCZOS
        ).copy()
    )

    image_original = np.array(image_original)
    detections = detections[0]
    obj_count = [0] * len(MASK)
    bboxs, cls_label = [], []
    if detections is not None:
        detections = rescale_boxes(detections, 512, img_orig.shape[:2])

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            obj_count[int(cls_pred)] += 1
            bboxs.append([int(x1), int(y1), int(x2), int(y2)])
            cls_label.append(int(cls_pred))

        text_statistic = "Найденные объекты:\n"
        for k, v in enumerate(obj_count):
            if v > 0:
                text_statistic += f"{MASK_RUS[k]} - {v}\n"
        text_statistic = text_statistic.rstrip("\n")
    else:
        text_statistic = "Ничего не найдено"

    return bboxs, cls_label, text_statistic


def draw_boxes(img, bboxs, cls_label):
    cmap = plt.get_cmap("tab20b")
    colors = [np.array(cmap(i)[:3]) * 255 for i in np.linspace(0, 1, 20)]
    unique_labels = np.unique(cls_label)
    n_cls_preds = unique_labels.shape[0]
    bbox_colors = random.sample(colors, n_cls_preds)

    max_orig_size = max(img.size)
    img_draw = np.array(
        img.resize((max_orig_size, max_orig_size),
                   Image.Resampling.LANCZOS).copy()
    )

    for i in range(len(bboxs)):
        x1, y1, x2, y2 = bboxs[i]
        color = bbox_colors[int(np.where(unique_labels == cls_label[i])[0])]
        img_draw = cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        img_draw = cv2.putText(
            img_draw,
            MASK_RUS[cls_label[i]],
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return img_draw
