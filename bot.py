import io
import telebot
import torch
import matplotlib

from telebot import types
from PIL import Image
from model.yolo import YoloV3

from eval.inference import model_inference
from tools.map import valid_coord, get_map_picture
from db.sql import MySQL

matplotlib.use('agg')


# Данные для входа в БД и токен Телеграмма
DATABASE_ADDRESS = "YOUR ADDRESS"
DATABASE_USER = "YOUR USER"
DATABASE_PASW = "YOUR PASSWORD"
DATABASE_NAME = "YOUR DATABASE NAME"
BOT_TOKEN = "YOUR TOKEN"

# Шаблон текста для кнопок бота
TEXT_HELLO = "Привет!👋 Это бот для обнаружения объектов на спутниковых снимках🌍🛰️"
TEXT_CLASSES = "Модель обучена на наборе данных [DIOR](https://arxiv.org/abs/1909.00133). Здесь содержится 20 различных классов\
	, которые могут быть на спутниковых снимках✈🚤🚞🏟🏭🌉"
TEXT_FEATURES = "Отправь мне картинку/скриншот со спутников🌍🛰️, программа попытается найти объекты на нем🔍 Можно воспользоваться [Google Earth](https://earth.google.com/web/), [Google Map](https://www.google.com/maps) или [Yandex Map](https://yandex.ru/maps)\n\nРекомендуется использовать Google Earth"
TEXT_MODEL = "Здесь используется одна из моделей Object Detection [YOLOv3](https://arxiv.org/abs/1804.02767)"
TEXT_EXAMPLE = "Демонстрация работы на некоторых снимках👁️ Репозиторий [GitHub](https://github.com/Vlad15lav/ors-detection) обученной модели"
TEXT_MAP = "Отправьте [координаты](https://www.latlong.net/) долготы, широты и zoom!\nZoom: [12-20]\n\nПример: 55.82103 49.16219 16"
TEXT_TOP = "Выберите какие данные интересуют📈"
# Разрешенные типы файлов для отправки
PICTURE_TYPES = ('png', 'jpg', 'jpeg')


# База данных
database = MySQL()
if not database.create_connection(DATABASE_ADDRESS, DATABASE_USER, DATABASE_PASW, DATABASE_NAME):
	raise "Ошибка в подключение MySQL!"

# Детектор YoloV3
model = YoloV3()
model.load_state_dict(torch.load('weights/dior_weights.pth', map_location=torch.device('cpu')))

# Телеграмм бот
bot = telebot.TeleBot(BOT_TOKEN)


# Событие кнопки Старт
@bot.message_handler(commands=['start'])
def handle_start_help(message):
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
	btn_question = types.KeyboardButton("Возможности❓")
	markup.add(btn_question)

	img_hello = open('imgs/hello_image.png', 'rb')
	bot.send_photo(message.chat.id, img_hello, caption=TEXT_HELLO, reply_markup=markup, parse_mode='Markdown')


# Событие отправки текста
@bot.message_handler(content_types=['text'])
def handle_message(message):
	img_classes = open('imgs/classes.png', 'rb')

	markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
	btn_classes = types.KeyboardButton("Объекты📰")
	btn_model = types.KeyboardButton("Модель⚙️")
	btn_example = types.KeyboardButton("Пример работыℹ️")
	btn_map = types.KeyboardButton("Быстрый скрин🔍")
	btn_question = types.KeyboardButton("Возможности❓")
	btn_stats = types.KeyboardButton("Статистика📈")
	markup.add(btn_classes, btn_model, btn_example, btn_map, btn_question, btn_stats)

	if message.text == "Объекты📰":
		# Узнать какие объекты находит модель
		bot.send_photo(message.chat.id, img_classes, caption=TEXT_CLASSES, reply_markup=markup, parse_mode='Markdown')
	elif message.text == "Возможности❓":
		# Информация о боте
		bot.send_message(message.chat.id, text=TEXT_FEATURES, reply_markup=markup, parse_mode='Markdown')

	elif message.text == "Модель⚙️":
		# Информаци о YoloV3
		bot.send_message(message.chat.id, text=TEXT_MODEL, reply_markup=markup, parse_mode='Markdown')

	elif message.text == "Пример работыℹ️":
		# Пример инференса для некоторых картинок
		img_example = open('imgs/preview.gif', 'rb')
		bot.send_video(message.chat.id, img_example, caption=TEXT_EXAMPLE, reply_markup=markup, parse_mode='Markdown')

	elif message.text == "Быстрый скрин🔍":
		# Взять скрин с API Yandex Map
		bot.send_message(message.chat.id, text=TEXT_MAP, reply_markup=markup, parse_mode='Markdown')

	elif message.text == "Статистика📈":
		# Узнать статистику по Боту
		markup_top = types.ReplyKeyboardMarkup(resize_keyboard=True)
		btn_top1 = types.KeyboardButton("Общая статистика")
		btn_top2 = types.KeyboardButton("Топ 5🏆")
		btn_top3 = types.KeyboardButton("Ваши данные")
		btn_top4 = types.KeyboardButton("Общая статистика за сутки")
		btn_top5 = types.KeyboardButton("Ваша статистика за сутки")

		markup_top.add(btn_top1, btn_top2, btn_top3, btn_top4, btn_top5)
		bot.send_message(message.chat.id, text=TEXT_TOP, reply_markup=markup_top, parse_mode='Markdown')

	elif message.text == "Общая статистика":
		text_query = database.get_popular_all_class()
		bot.send_message(message.chat.id, text=text_query, reply_markup=markup)

	elif message.text == "Топ 5🏆":
		text_query = database.get_top_user()
		bot.send_message(message.chat.id, text=text_query, reply_markup=markup)

	elif message.text == "Ваши данные":
		text_query = database.get_count_user_class(message.from_user.username)
		bot.send_message(message.chat.id, text=text_query, reply_markup=markup)

	elif message.text == "Общая статистика за сутки":
		text_query = database.get_popular_all_last()
		bot.send_message(message.chat.id, text=text_query, reply_markup=markup)

	elif message.text == "Ваша статистика за сутки":
		text_query = database.get_count_user_last(message.from_user.username)
		bot.send_message(message.chat.id, text=text_query, reply_markup=markup)

	else:
		# Получаем координаты с Yandex API
		coord = valid_coord(message.text)

		if not coord is None:
			# Картинка с Yandex API
			imageFile = get_map_picture(*coord)
			if not imageFile is None:
				# Поиск объектов
				model_result, text_statistic, user_stats = model_inference(model, imageFile)
				if len(user_stats) > 0:
					database.add_request(message.from_user.username, user_stats.keys(), user_stats.values())
				
				bot.send_photo(message.chat.id, model_result, caption=text_statistic, reply_markup=markup)
			else:
				bot.send_message(message.chat.id, text="Некорректные координаты!", reply_markup=markup, parse_mode='Markdown')
		else:
			bot.send_message(message.chat.id, text="Некорректные координаты!", reply_markup=markup, parse_mode='Markdown')


# Событие отправки фото
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
	btn_classes = types.KeyboardButton("Объекты📰")
	btn_model = types.KeyboardButton("Модель⚙️")
	btn_example = types.KeyboardButton("Пример работыℹ️")
	btn_map = types.KeyboardButton("Быстрый скрин🔍")
	btn_question = types.KeyboardButton("Возможности❓")
	btn_stats = types.KeyboardButton("Статистика📈")
	markup.add(btn_classes, btn_model, btn_example, btn_map, btn_question, btn_stats)

	bot.send_message(message.chat.id, text="Начинаю поиск объектов⌛", reply_markup=markup, parse_mode='Markdown')

	# Скачиваем файл
	fileID = message.photo[-1].file_id
	file_info = bot.get_file(fileID)
	downloaded_file = bot.download_file(file_info.file_path)

	imageStream = io.BytesIO(downloaded_file)
	imageFile = Image.open(imageStream)
	#imageStream.close()

	# Поиск объектов
	model_result, text_statistic, user_stats = model_inference(model, imageFile)
	if len(user_stats) > 0:
		database.add_request(message.from_user.username, user_stats.keys(), user_stats.values())

	bot.send_photo(message.chat.id, model_result, caption=text_statistic, reply_markup=markup)


# Событие отправки фото как файла
@bot.message_handler(content_types=['document'])
def handle_docs(message):
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
	btn_classes = types.KeyboardButton("Объекты📰")
	btn_model = types.KeyboardButton("Модель⚙️")
	btn_example = types.KeyboardButton("Пример работыℹ️")
	btn_map = types.KeyboardButton("Быстрый скрин🔍")
	btn_question = types.KeyboardButton("Возможности❓")
	btn_stats = types.KeyboardButton("Статистика📈")
	markup.add(btn_classes, btn_model, btn_example, btn_map, btn_question, btn_stats)
	
	bot.send_message(message.chat.id, text="Начинаю поиск объектов⌛", reply_markup=markup, parse_mode='Markdown')
	try:
		chat_id = message.chat.id

		# Проверяем тип файла
		file_info = bot.get_file(message.document.file_id)
		file_type = file_info.file_path.split('.')[-1].lower()

		if file_type in PICTURE_TYPES:
			# Скачиваем файл пользователя
			downloaded_file = bot.download_file(file_info.file_path)

			imageStream = io.BytesIO(downloaded_file)
			imageFile = Image.open(imageStream)
			imageStream.close()

			# Поиск объектов
			model_result, text_statistic, user_stats = model_inference(model, imageFile)
			if len(user_stats) > 0:
				database.add_request(message.from_user.username, user_stats.keys(), user_stats.values())

			bot.send_photo(message.chat.id, model_result, caption=text_statistic, reply_markup=markup, parse_mode='Markdown')
		else:
			bot.send_message(message.chat.id, text="Отправьте файл JPG, PNG!", reply_markup=markup)

	except Exception as e:
		bot.send_message(message.chat.id, text=f"Ошибка отправки!", reply_markup=markup)


if __name__ == '__main__':
	bot.polling(none_stop=True)