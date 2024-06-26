"""
Файл-заготовка под создание графического наглядного редактора изображений

Программа должна открывать произвольное изображение и позволять интерактивно его редактировать для наглядного
отображения возможностей Аугментатора.

В процессе разработки
"""

from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Slider, RadioButtons, CheckButtons


def updateGraph():
    """!!! Функция для обновления графика"""
    global slider_x_range
    global slider_y_range
    global graph_axes
    global x_min, x_max, y_min, y_max
    global radiobuttons
    global checkbuttons_grid

    options = {"Opt1": "r", "Opt2": "b", "Opt3": "g"}

    # Получаем значения интервалов
    x_min, x_max = slider_x_range.val
    y_min, y_max = slider_y_range.val

    # Загрузим изображение
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Файл {input_path} не найден")

    # Конвертируем изображение в RGB формат для корректного отображения с помощью matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    actions = options[radiobuttons.value_selected]

    graph_axes.clear()
    graph_axes.imshow(image)
    graph_axes.set_xlim(x_min, x_max)
    graph_axes.set_ylim(y_max, y_min)  # Инвертируем ось Y для корректного отображения

    # !!! Определим, нужно ли показывать сетку на графике
    grid_visible = checkbuttons_grid.get_status()[0]
    graph_axes.grid(grid_visible)

    plt.draw()


def onCheckClicked(value: str):
    """!!! Обработчик события при нажатии на флажок"""
    updateGraph()


def onRadioButtonsClicked(value: str):
    """Обработчик события при клике по RadioButtons"""
    updateGraph()


def onChangeValue(value: np.float64):
    """Обработчик события изменения значений μ и σ"""
    updateGraph()


def onChangeXRange(value: Tuple[np.float64, np.float64]):
    """Обработчик события измерения значения интервала по оси X"""
    updateGraph()


def onChangeYRange(value: Tuple[np.float64, np.float64]):
    """Обработчик события измерения значения интервала по оси Y"""
    updateGraph()


if __name__ == "__main__":
    # Путь к изображению
    input_path = 'test1.jpg'  # Замените на путь к вашему изображению

    # Начальные параметры графиков
    x_min = 0
    x_max = 1024  # Ширина изображения
    y_min = 0
    y_max = 1024  # Высота изображения

    # Создадим окно с графиком
    fig, graph_axes = plt.subplots()

    # Выделим область, которую будет занимать график
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)

    # Создадим слайдер для задания интервала по оси X
    axes_slider_x_range = plt.axes([0.3, 0.09, 0.5, 0.04])
    slider_x_range = RangeSlider(
        axes_slider_x_range,
        label="x",
        valmin=0.0,
        valmax=1024.0,
        valinit=(x_min, x_max),
        valfmt="%1.0f",
    )

    # Создадим слайдер для задания интервала по оси Y
    axes_slider_y_range = plt.axes([0.3, 0.03, 0.5, 0.04])
    slider_y_range = RangeSlider(
        axes_slider_y_range,
        label="y",
        valmin=0.0,
        valmax=1024.0,
        valinit=(y_min, y_max),
        valfmt="%1.0f",
    )

    # Создадим оси для переключателей
    axes_radiobuttons = plt.axes([0.05, 0.09, 0.17, 0.2])

    # Создадим переключатель
    radiobuttons = RadioButtons(
        axes_radiobuttons, ["Opt1", "Opt2", "Opt3"]
    )

    # !!! Создадим оси для флажка
    axes_checkbuttons = plt.axes([0.05, 0.01, 0.17, 0.07])

    # !!! Создадим флажок
    checkbuttons_grid = CheckButtons(axes_checkbuttons, ["Сетка"], [True])

    # Подпишемся на события при изменении значения слайдеров.
    slider_x_range.on_changed(onChangeXRange)
    slider_y_range.on_changed(onChangeYRange)

    # Подпишемся на событие при переключении радиокнопок
    radiobuttons.on_clicked(onRadioButtonsClicked)

    # !!! Подпишемся на событие при клике по флажку
    checkbuttons_grid.on_clicked(onCheckClicked)

    updateGraph()
    plt.show()
