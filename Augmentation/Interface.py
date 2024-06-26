import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, CheckButtons, TextBox, Button
from logger import logger

def convert_data_to_random_params(data):
    """
    Преобразует данные формата интерфейса в формат словаря класса Augmentation.Images

    :param data: Набор специфических данных, полученных из функции on_button_clicked описанной ниже.
    :return: Словарь формата класса Images.random_params.
    """
    try:
        multy = int(data['multy'])
    except:
        multy = 1

    random_params = {
        'multiplicator': multy,
        'path_in' : data['directory_in'],
        'path_out': data['directory_out'],
        'blur': {
            'enable': data['checkboxes'][0] or data['checkboxes'][1],
            'power_x': data['sliders'][0],
            'power_y': data['sliders'][1]
        },
        'brightness': {
            'enable': data['checkboxes'][2],
            'range': data['sliders'][2]
        },
        'flip': {
            'enable': any(data['check_buttons']),
            'flip_code': (
                -1 if data['check_buttons'][0] else None,
                0 if data['check_buttons'][1] else None,
                1 if data['check_buttons'][2] else None,
            )
        },
        'saturation': {
            'enable': data['checkboxes'][3],
            'range': data['sliders'][3]
        },
        'noise': {
            'enable': data['checkboxes'][4] or data['checkboxes'][5],
            'mean_range': data['sliders'][4],
            'variance_range': data['sliders'][5]
        },
        'contrast': {
            'enable': data['checkboxes'][6],
            'range': data['sliders'][6]
        },
        'crop': {
            'enable': any(data['checkboxes'][7:11]),
            'random': False,
            'left': data['sliders'][7],
            'top': data['sliders'][8],
            'window_width': data['sliders'][9],
            'window_height': data['sliders'][10]
        },
        'resize': {
            'enable': any(data['checkboxes'][11:13]),
            'width_range': data['sliders'][11],
            'height_range': data['sliders'][12]
        },
        'rotate': {
            'enable': data['checkboxes'][13],
            'angle_range': data['sliders'][13]
        },
        'text': {
            'enable': any(data['checkboxes'][14:]),
            'text': data['text'],
            'position_x_range': data['sliders'][14],
            'position_y_range': data['sliders'][15],
            'font_scale_range': data['sliders'][16],
            'color_range': (data['sliders'][18], data['sliders'][19], data['sliders'][20]),
            'enable_blur': data['checkboxes'][17],
            'blur_range': (data['sliders'][21],data['sliders'][22]),
            'angle_range': data['sliders'][17]
        }
    }
    return random_params

def create_gui(num = None):
    """
    Создаёт графический интерфейс

    :param num: Номер папки вывода - необходимо для многократного использования интерфейса, для отсутствия
     перезаписи изображений при отсутствии изменения папки ввода.
    :return: None
    """

    global sliders, checkboxes, check_buttons, text_dir_in_box, text_dir_out_box, button, text_multy_box, text_in_box


    sliders = []     # Список слайдеров RangeSlider
    checkboxes = []  # Список для хранения всех чек-боксов

    names = [ # Названия для всех слайдеров
        'Blur power x',
        'Blur power y',
        'Brightness',
        'Saturation',
        'Mean of noise',
        'Var of noise',
        'Contrast',
        'Crop left',
        'Crop top',
        'Crop window width',
        'Crop window height ',
        'Resize width',
        'Resize height',
        'Rotate angle',
        'Text x position',
        'Text y position',
        'Text font scale (in tenth)',
        'Text rotation',
        'Text color red',
        'Text color Blue',
        'Text color Green',
        'Text blur power x',
        'Text blur power y',
    ]

    vals = [ # Интервал для всех слайдеров
        (0, 25),  # 'Blur power x',
        (0, 25),  # 'Blur power y',
        (50, 200),  # 'Brightness',
        (50, 200),  # 'Saturation',
        (0, 50),  # 'Mean of noise',
        (0, 50),  # 'Var of noise',
        (50, 200),  # 'Contrast',
        (0, 1024),  # 'Crop left',
        (0, 1024),  # 'Crop top',
        (1, 1024),  # 'Crop window width',
        (1, 1024),  # 'Crop window height ',
        (1, 1024),  # 'Resize width',
        (1, 1024),  # 'Resize height',
        (-20, 20),  # 'Rotate angle',
        (0, 1024),  # 'Text x position',
        (0, 1024),  # 'Text y position',
        (1, 50),  # 'Text font scale (in tenth)',
        (-15, 15),  # 'Text rotation',
        (0, 255),  # 'Text color red',
        (0, 255),  # 'Text color Blue',
        (0, 255),  # 'Text color Green',
        (0, 25),  # 'Text blur power x',
        (0, 25),  # 'Text blur power y',
    ]

    start = [ # Начальные значения слайдеров
        (5, 10),  # 'Blur power x',
        (5, 10),  # 'Blur power y',
        (75, 150),  # 'Brightness',
        (75, 150),  # 'Saturation',
        (16, 32),  # 'Mean of noise',
        (16, 32),  # 'Var of noise',
        (75, 150),  # 'Contrast',
        (128, 512),  # 'Crop left',
        (128, 512),  # 'Crop top',
        (128, 512),  # 'Crop window width',
        (128, 512),  # 'Crop window height ',
        (512, 1024),  # 'Resize width',
        (512, 1024),  # 'Resize height',
        (-5, 5),  # 'Rotate angle',
        (256, 386),  # 'Text x position',
        (256, 386),  # 'Text y position',
        (5, 15),  # 'Text font scale (in tenth)',
        (-5, 5),  # 'Text rotation',
        (0, 255),  # 'Text color red',
        (0, 255),  # 'Text color Blue',
        (0, 255),  # 'Text color Green',
        (3, 8),  # 'Text blur power x',
        (3, 8),  # 'Text blur power y',
    ]

    # Создаем слайдеры и чек-боксы
    for i in range(len(names)):
        # Создаём слайдеры и подписываем их
        ax_slider = plt.axes([0.1, 0.9 - i * 0.03, 0.55, 0.03])
        slider = RangeSlider(ax_slider, names[i], *vals[i], valinit=start[i], valstep=1)
        sliders.append(slider)

        # Создаем чек-боксы справа от слайдеров
        ax_checkbox = plt.axes([0.72, 0.9 - i * 0.03, 0.015, 0.03], frame_on=False)
        checkbox = CheckButtons(ax_checkbox, [''], actives=[False])  # Пустой чек-бокс
        checkboxes.append(checkbox)

    # Создаем CheckButtons для вариантов отзеркаливания
    check_ax = plt.axes([0.8, 0.3, 0.1, 0.1])
    check_labels = ['Horizontal flip', 'No flip', 'Vertical flip']
    check_buttons = CheckButtons(check_ax, check_labels, actives=[True, True, True])

    # Создаем TextBox для ввода адреса ввода
    text_dir_in_ax = plt.axes([0.85, 0.9, 0.1, 0.05])
    text_dir_in_box = TextBox(text_dir_in_ax, 'Directory In', initial='./input/')

    # Создаем TextBox для ввода адреса вывода
    text_dir_out_ax = plt.axes([0.85, 0.8, 0.1, 0.05])
    text_dir_out_box = TextBox(text_dir_out_ax, 'Directory Out', initial=f'./output/out{num if num else ""}')

    # Создаем TextBox для ввода количества изображений
    text_multy_ax = plt.axes([0.85, 0.7, 0.1, 0.05])
    text_multy_box = TextBox(text_multy_ax, 'Multy', initial='1')

    # Создаем TextBox для ввода текста
    text_in_ax = plt.axes([0.8, 0.6, 0.19, 0.05])
    text_in_box = TextBox(text_in_ax, 'Text', initial='Hello world!')

    # Создаем кнопку "Принять"
    button_ax = plt.axes([0.8, 0.1, 0.1, 0.05])
    button = Button(button_ax, 'Принять')

    button.on_clicked(on_button_clicked)

    logger.info(f'Интерфейс успешно создан!')

    plt.show()

def on_button_clicked(event = None):
    """

    :param event: Произошедшее событие
    :return: словарь формата Augmentation.Images.random_params
    """

    global data_dict

    # Собираем значения слайдеров
    slider_values = [slider.val for slider in sliders]

    # Собираем значения чек-боксов
    checkbox_values = [checkbox.get_status()[0] for checkbox in checkboxes]

    # Получаем состояние CheckButtons
    check_states = check_buttons.get_status()

    # Получаем адрес директории ввода
    dir_in_value = text_dir_in_box.text

    multy = text_multy_box.text

    # Получаем адрес директории вывода
    dir_out_value = text_dir_out_box.text

    # Заносим данные в словарь
    data_dict = {
        'sliders': slider_values,
        'checkboxes': checkbox_values,
        'check_buttons': check_states,
        'directory_in': dir_in_value,
        'directory_out': dir_out_value,
        'multy' : multy,
        'text' : text_in_box.text,
    }

    # Преобразуем данные в случайные параметры и выводим результат
    random_params = convert_data_to_random_params(data_dict)

    # Закрываем окно matplotlib
    plt.close()
    logger.info('Параметры сохранены')
    return random_params


if __name__ == '__main__':
    create_gui()