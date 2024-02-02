import folium
import os
import streamlit as st
import yaml
import wget

from streamlit_folium import st_folium
from PIL import Image

import torch

from model.yolo import YoloV3
from tools.inference import model_inference, draw_boxes
from tools.map import get_map_picture


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def load_weights():
    if not os.path.exists("states/dior_weights.pth"):
        if not os.path.exists("states"):
            os.mkdir("states")
        wget.download(
            "https://github.com/Vlad15lav/" /
            "ors-detection/releases/download/weights/dior_weights.pth",
            out="states/dior_weights.pth",
        )


@st.cache_data
def load_model():
    cfg = Params("projects/dior.yml")

    model = YoloV3(len(cfg.mask), cfg.anchors, 512)
    try:
        model.load_state_dict(
            torch.load("states/dior_weights.pth",
                       map_location=torch.device("cpu"))
        )
    except FileNotFoundError:
        print(
            "Weights is not found. You should move the weights to \
            /states/{name_proj}_weights.pth"
        )

    return model


# def get_inference(img, model):
#     bbox, cls_label, obj_count = model_inference(model, img)
#     img_draw = draw_boxes(img, bbox, cls_label)


def main():
    st.set_page_config(page_title="Space Detector",
                       page_icon="🌍",
                       layout="centered")

    st.title("AI Space Detector")
    st.write("Сервис для нахождения объектов на спутниковых снимках.")
    with st.expander("Подробнее о приложении"):
        st.write(
            "Привет!👋 Это сервис для обнаружения объектов " /
            "на спутниковых снимках🌍🛰️."
        )
        st.write(
            "Модель обучена на наборе данных [DIOR]" /
            "(https://arxiv.org/abs/1909.00133). " /
            "Здесь содержится 20 различных классов, " /
            "которые могут быть на спутниковых снимках✈🚤🚞🏟🏭🌉."
        )
        st.image("./images/classes.png")
        st.write(
            "Отправь мне картинку/скриншот со спутников🌍🛰️, " /
            "программа попытается найти объекты на нем🔍. " /
            "Можно воспользоваться [Google Earth]" /
            "(https://earth.google.com/web/), [Google Map]" /
            "(https://www.google.com/maps) или [Yandex Map]" /
            "(https://yandex.ru/maps).\n" /
            "Рекомендуется использовать Google Earth."
        )
        st.write(
            "Здесь используется одна из моделей Object Detection " /
            "[YOLOv3](https://arxiv.org/abs/1804.02767)."
        )
        st.write(
            "Репозиторий [GitHub]" /
            "(https://github.com/Vlad15lav/ors-detection) обученной модели."
        )

    with st.sidebar:
        latitude = st.sidebar.text_input("Широта", "55.82103")
        longitude = st.sidebar.text_input("Долгата", "49.16219")
        zoom = st.sidebar.text_input("Масштаб", "16")

        conf = st.slider("Уверенность классификатора", 0.01, 1.0, 0.35)
        select_region = st.button("Отправить координаты")

    uploaded_file = st.file_uploader("Загрузите фото",
                                     type=["png", "jpg", "jpeg"])
    load_weights()
    model = load_model()

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                "<div style='text-align: center;'>" /
                f"{'Оригинальное изображение'}</div>",
                unsafe_allow_html=True,
            )
            # st.write('Оригинальное изображение')
            st.image(uploaded_file)
            image = Image.open(uploaded_file)

        bbox, cls_label, text_statistic = model_inference(model,
                                                          image,
                                                          conf_tresh=conf)
        img_draw = draw_boxes(image, bbox, cls_label)

        with col2:
            # st.write('Предсказание модели')
            st.markdown(
                "<div style='text-align: center;'>" /
                f"{'Предсказание модели'}</div>",
                unsafe_allow_html=True,
            )
            st.image(img_draw)
            st.text(text_statistic)

    if select_region:
        uploaded_file = get_map_picture(latitude, longitude, zoom)

        if uploaded_file is None:
            st.write("Не удалось получить спутниковое изображени! " /
                     "Ошибка в координатах")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    "<div style='text-align: center;'>" /
                    f"{'Оригинальное изображение'}</div>",
                    unsafe_allow_html=True,
                )
                st.image(uploaded_file)

            bbox, cls_label, text_statistic = model_inference(
                model, uploaded_file, conf_tresh=conf
            )
            img_draw = draw_boxes(uploaded_file, bbox, cls_label)

            with col2:
                st.markdown(
                    "<div style='text-align: center;'>" /
                    f"{'Предсказание модели'}</div>",
                    unsafe_allow_html=True,
                )
                st.image(img_draw)
                st.text(text_statistic)

            m = folium.Map(location=[latitude, longitude], zoom_start=zoom)
            folium.Marker(
                [latitude, longitude],
                popup="Текущие координаты",
                tooltip=f"Текущие координаты: {longitude} {latitude}",
            ).add_to(m)
            st_folium(m, width=725, returned_objects=[])


if __name__ == "__main__":
    main()
