import io
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from PIL import Image
from torchvision import transforms
from matplotlib.backends.backend_agg import FigureCanvas
from model.yolo import YoloV3
from tools.nms import non_max_suppression
from tools.box import rescale_boxes

def model_inference(model, image_original, conf_tresh: float=0.35):
	mask = ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
		'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
		'golffield', 'groundtrackfield', 'harbor', 'ship', 'stadium', 'storagetank',
		'tenniscourt', 'trainstation', 'vehicle', 'windmill', 'overpass']
	mask_rus = ['Самолет', 'Аэропорт', 'Бейсбольная площадка', 'Баскетбольная площадка',
				'Мост', 'Завод', 'Дамба', 'Зона обслуживания авто', 'Пропускной пунтк',
				'Площадка для гольфа', 'Поле для бега', 'Гавань', 'Корабль', 'Стадион',
				'Резервуары хранения', 'Теннисный корт', 'ЖД станция', 'Машина',
				'Ветрогенератор', 'Эстакада']
	user_classes = {}

	# Ресайз фото под вход модели
	image = image_original.copy()
	image = image.resize((512, 512), Image.Resampling.LANCZOS)

	# Задаем каналы
	image = transforms.ToTensor()(image.convert('RGB'))
	if len(image.shape) != 3:
		image = image.unsqueeze(0)
		image = image.expand((3, image.shape[1:]))

	# Предсказание модели и NMS для удаление перекрывающих боксов
	predict = model(image[None, :])
	detections = non_max_suppression(predict, conf_thres=conf_tresh, nms_thres=0.5)

	#
	max_orig_size = max(image_original.size)
	img_orig = np.array(image_original.resize((max_orig_size, max_orig_size), Image.Resampling.LANCZOS).copy())
	
	fig, ax = plt.subplots(1)
	ax.axis('off')
	ax.imshow(img_orig)

	cmap = plt.get_cmap("tab20b")
	colors = [cmap(i) for i in np.linspace(0, 1, 20)]
	
	detections = detections[0]
	obj_count = [0] * len(mask)
	if detections is not None:
		detections = rescale_boxes(detections, 512, img_orig.shape[:2])
		unique_labels = detections[:, -1].cpu().unique()
		n_cls_preds = len(unique_labels)
		bbox_colors = random.sample(colors, n_cls_preds)
		
		for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
			obj_count[int(cls_pred)] += 1
			box_w = x2 - x1
			box_h = y2 - y1

			color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
			# Создайте прямоугольник патча
			bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
			# Добавить вставку в график
			ax.add_patch(bbox)
			# Добавить метку класса
			plt.text(
				x1,
				y1,
				s=mask[int(cls_pred)],
				color="white",
				verticalalignment="top",
				bbox={"color": color, "pad": 0},
			)

			if user_classes.get(int(cls_pred) + 1, False):
				user_classes[int(cls_pred) + 1] += 1
			else:
				user_classes[int(cls_pred) + 1] = 1

		text_statistic = "Найденные объекты:\n"
		for k, v in enumerate(obj_count):
			if v > 0:
				text_statistic += f"{mask_rus[k]} - {v}\n"
		text_statistic = text_statistic.rstrip('\n')
	else:
		text_statistic = "Ничего не найдено"


	plt.subplots_adjust(wspace=0, hspace=0)
	plt.tight_layout(pad=0)

	data = io.BytesIO()
	plt.savefig(data, format='png', dpi=400)
	data.seek(0)

	return data, text_statistic, user_classes