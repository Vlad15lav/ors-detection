import mysql.connector

from datetime import datetime, timedelta
from mysql.connector import Error

class MySQL:
	def __init__(self):
		self.connection = None
		self.host_name = None
		self.user_name = None
		self.user_password = None
		self.databases = None

	def create_connection(self, host_name: str,
								user_name: str,
								user_password: str,
								databases: str) -> bool:
		flag = True
		self.host_name = host_name
		self.user_name = user_name
		self.user_password = user_password
		self.databases = databases

		try:
			self.connection = mysql.connector.connect(
				host=host_name,
				user=user_name,
				passwd=user_password,
				database=databases
			)
# 			self.connection.execute('set max_allowed_packet=67108864')
		except Error as e:
			flag = False

		return flag

	def stop_connection(self):
		self.connection.close()

	def select(self, query: str):
		try:
			self.connection.commit()
			if self.connection is None:
				return False
			with self.connection.cursor() as cursor:
				cursor.execute(query)
				result = cursor.fetchall()
				return result
		except Exception as e:
			if not self.create_connection(self.host_name,
											self.user_name,
											self.user_password,
											self.database):
				assert f"MySQL connection fail! {e}"

			self.connection.commit()
			if self.connection is None:
				return False
			with self.connection.cursor() as cursor:
				cursor.execute(query)
				result = cursor.fetchall()
				return result


	def insert(self, query: str, value: tuple):
		if self.connection is None:
			return False

		try:
			with self.connection.cursor() as cursor:
				cursor.execute(query, value)
			self.connection.commit()

		except Exception as e:
			if not self.create_connection(self.host_name,
											self.user_name,
											self.user_password,
											self.database):
				assert f"MySQL connection fail! {e}"

			with self.connection.cursor() as cursor:
				cursor.execute(query, value)
			self.connection.commit()


	def add_request(self, user, id_cls, counts):
		id_user = self.select(f"SELECT id_user FROM users WHERE user_name = '{user}'")

		# нужно добавить нового пользователя
		if len(id_user) == 0:
			self.insert("INSERT INTO users (user_name) VALUES (%s)", (user, ))
			id_user = self.select(f"SELECT id_user FROM users WHERE user_name = '{user}'")
		id_user = id_user[0][0]

		id_cls = list(id_cls)
		counts = list(counts)
		for i in range(len(id_cls)):
			self.insert("INSERT INTO requests (id_user, id_cls, count, date) VALUES (%s, %s, %s, %s)",
				(id_user, id_cls[i], counts[i], datetime.now()))

	# Самые популярные классы среди пользователей
	def get_popular_all_class(self):
		qr_text = "SELECT c.class_name, SUM(r.count) AS sum_count \
					FROM requests AS r JOIN classes AS c USING(id_cls) \
					GROUP BY c.class_name \
					ORDER BY sum_count DESC"

		result = self.select(qr_text)
		if len(result):
			text = "Статистика по найденным объектом среди всех пользователей:\n\n"
			for row in result:
				text += f"{row[0]} - {row[1]}\n"
			return text.rstrip('\n')
		else:
			return "Данные отсутствуют!"

	# Сколько найденных объектов пользователем
	def get_count_user_class(self, user_name):
		qr_text = f"SELECT c.class_name, SUM(r.count) AS sum_count \
					FROM requests AS r JOIN classes AS c USING(id_cls) \
					JOIN users AS u USING(id_user) \
					WHERE u.user_name = '{user_name}' \
					GROUP BY c.class_name \
					ORDER BY sum_count DESC"

		result = self.select(qr_text)
		if len(result):
			text = f"Статистика по найденным объектам {user_name}:\n\n"
			for row in result:
				text += f"{row[0]} - {row[1]}\n"
			return text.rstrip('\n')
		else:
			return "Данные отсутствуют!"

	# Сколько объектов нашли за последние 24 часа
	def get_popular_all_last(self):
		qr_text = f"SELECT c.class_name, SUM(r.count) AS sum_count \
					FROM requests AS r JOIN classes AS c USING(id_cls) \
					WHERE r.date > NOW() - INTERVAL 1 DAY \
					GROUP BY c.class_name \
					ORDER BY sum_count DESC"

		result = self.select(qr_text)
		if len(result):
			text = f"Статистика по найденным объектам среди всех пользователей за сутки:\n\n"
			for row in result:
				text += f"{row[0]} - {row[1]}\n"
			return text.rstrip('\n')
		else:
			return "Данные отсутствуют!"

	# Сколько объектов нашел пользователь за последние 24 часа
	def get_count_user_last(self, user_name):
		qr_text = f"SELECT c.class_name, SUM(r.count) AS sum_count \
					FROM requests AS r JOIN classes AS c USING(id_cls) \
					JOIN users AS u USING(id_user) \
					WHERE r.date > NOW() - INTERVAL 1 DAY AND u.user_name = '{user_name}' \
					GROUP BY c.class_name \
					ORDER BY sum_count DESC"

		result = self.select(qr_text)
		if len(result):
			text = f"Статистика по найденным объектам {user_name} за сутки:\n\n"
			for row in result:
				text += f"{row[0]} - {row[1]}\n"
			return text.rstrip('\n')
		else:
			return "Данные отсутствуют!"

	# Топ 5 пользователей по найденным объектам
	def get_top_user(self):
		qr_text = "SELECT u.user_name, SUM(r.count) AS sum_count \
					FROM requests AS r JOIN users AS u USING(id_user) \
					GROUP BY u.user_name \
					ORDER BY sum_count DESC \
					LIMIT 5"

		result = self.select(qr_text)
		if len(result):
			text = "Топ пользователей по найденным объектам:\n\n"
			for row in result:
				text += f"{row[0]} - {row[1]}\n"
			return text.rstrip('\n')
		else:
			return "Данные отсутствуют!"


"""
База данных detector-bot

Таблицы:
	classes (id_cls, class_name)
	requests (id_req, id_user, id_cls, count, datetime)
	users (id_user, user_name)

CREATE TABLE classes (
	 id_cls INT NOT NULL AUTO_INCREMENT,
	 class_name CHAR(30) NOT NULL,
	 PRIMARY KEY (id_cls)
);

CREATE TABLE users (
	 id_user INT NOT NULL AUTO_INCREMENT,
	 user_name CHAR(50) NOT NULL,
	 PRIMARY KEY (id_user)
);

CREATE TABLE requests (
	 id_req INT NOT NULL AUTO_INCREMENT,
	 id_user INT NOT NULL,
	 id_cls INT NOT NULL,
	 count INT NOT NULL,
	 date DATETIME,
	 PRIMARY KEY (id_req)
);

INSERT INTO classes (class_name) VALUES ('airplane'), ('airport'), ('baseballfield'), ('basketballcourt'), ('bridge'),
		('chimney'), ('dam'), ('Expressway-Service-area'), ('Expressway-toll-station'),
		('golffield'), ('groundtrackfield'), ('harbor'), ('ship'), ('stadium'), ('storagetank'),
		('tenniscourt'), ('trainstation'), ('vehicle'), ('windmill'), ('overpass');

Запросы:
	1) Самые популярные классы среди пользователей

	SELECT c.class_name, SUM(r.count) AS sum_count
	FROM requests AS r JOIN classes AS c USING(id_cls)
	GROUP BY c.class_name
	ORDER BY sum_count;

	2) Сколько найденных объектов пользователем

	SELECT c.class_name, SUM(r.count) AS sum_count
	FROM requests AS r JOIN classes AS c USING(id_cls)
	JOIN users AS u USING(id_user)
	WHERE u.user_name = "vlad_15lav"
	GROUP BY c.class_name
	ORDER BY sum_count DESC;

	3) Сколько объектов нашли за последние 24 часа

	SELECT c.class_name, SUM(r.count) AS sum_count
	FROM requests AS r JOIN classes AS c USING(id_cls)
	WHERE r.date > NOW() - INTERVAL 1 DAY
	GROUP BY c.class_name
	ORDER BY sum_count DESC;

	4) Сколько объектов нашел пользователь за последние 24 часа

	SELECT c.class_name, SUM(r.count) AS sum_count
	FROM requests AS r JOIN classes AS c USING(id_cls)
	WHERE r.date > NOW() - INTERVAL 1 DAY
	AND u.user_name = "vlad_15lav"
	GROUP BY c.class_name
	ORDER BY sum_count DESC;

	5) Топ 5 пользователей по найденным объектам

	SELECT u.user_name, SUM(r.count) AS sum_count
	FROM requests AS r JOIN users AS u USING(id_user)
	WHERE r.date > NOW() - INTERVAL 1 DAY
	GROUP BY u.user_name
	ORDER BY sum_count DESC
	LIMIT 5;

"""

