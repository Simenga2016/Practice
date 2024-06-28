import random
from copy import deepcopy
from logger import logger


def apply_blur(img, params):
    """
    Добавляет размытие.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        power_x = random.randint(*params.get('power_x', (11, 11)))
        power_y = random.randint(*params.get('power_y', (11, 11)))
        img.blur_image(power_x, power_y)


def apply_brightness(img, params):
    """
    Добавляет яркость.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        brightness = random.randint(*params.get('range', (100, 100))) / 100
        img.change_brightness(brightness)


def apply_flip(img, params):
    """
    Отзеркаливает изображение.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        flip = random.choice([x for x in params.get('flip_code', [0, ]) if x is not None])
        if flip == -1:
            img.flip_horizontal()
        elif flip == 1:
            img.flip_vertical()


def apply_saturation(img, params):
    """
    Добавляет насыщенность.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        img.change_saturation(random.randint(*params.get('range', (50, 50))) / 100)


def apply_noise(img, params):
    """
    Добавляет шум.

    :param img: Исходное изображение 
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        var = random.randint(*params.get('variance_range', (4, 4)))
        mean = random.randint(*params.get('mean_range', (4, 4)))
        img.add_noise(var, mean)


def apply_contrast(img, params):
    """
    Добавляет контрастность.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        img.change_contrast(random.randint(*params.get('range', (100, 100))) / 100)


def apply_crop(img, params):
    """
    Обрезает изображение.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        tmp_img = deepcopy(img)
        while not (tmp_img is not None and tmp_img.image.size > 0):
            x = random.randint(*params.get('left', (0, 0)))
            y = random.randint(*params.get('top', (0, 0)))
            wid = random.randint(*params.get('window_width', (0, 0)))
            hei = random.randint(*params.get('window_height', (0, 0)))
            if params.get('random', False):
                tmp_img.random_crop_image(wid, hei, x, y)
            else:
                tmp_img.crop_image(x, y, wid, hei)
        img = deepcopy(tmp_img)
    return img


def apply_resize(img, params):
    """
    Изменяет размер.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        wid = random.randint(*params.get('width_range', (512, 512)))
        hei = random.randint(*params.get('height_range', (512, 512)))
        img.resize_image(wid, hei)


def apply_rotate(img, params):
    """
    Поварачивает изображение.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        img.rotate_image(random.randint(*params.get('angle_range', (0, 0))))


def apply_text(img, params):
    """
    Добавляет текст.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    if params.get('enable', False):
        text = params.get('text', ' ')
        for txt in text.split('\n'):
            position = (
                random.randint(*params.get('position_x_range', (0, 0))),
                random.randint(*params.get('position_y_range', (0, 0)))
            )
            scale = random.randint(*params.get('font_scale_range', (10, 10))) / 10
            color_tmp = params.get('color_range', ((255, 255), (255, 255), (255, 255)))
            color = (
                random.randint(*color_tmp[0]),
                random.randint(*color_tmp[1]),
                random.randint(*color_tmp[2])
            )
            thick = random.randint(*params.get('thickness_range', (1, 1)))
            blur = params.get('enable_blur', False)
            power_tmp = params.get('blur_range', ((3, 37), (3, 37)))
            power = (
                random.randint(*power_tmp[0]),
                random.randint(*power_tmp[1])
            )
            angle = random.randint(*params.get('angle_range', (-15, 15)))

            img.add_text_to_image(txt, position, scale, color, thick, blur, power, angle)


def apply_augmentations(img, params):
    """
    Применяет все указанные модификации.

    :param img: Исходное изображение
    :param params: Параметры
    :return: None
    """
    apply_blur(img, params.get('blur', {}))
    apply_brightness(img, params.get('brightness', {}))
    apply_flip(img, params.get('flip', {}))
    apply_saturation(img, params.get('saturation', {}))
    apply_noise(img, params.get('noise', {}))
    apply_contrast(img, params.get('contrast', {}))
    img = apply_crop(img, params.get('crop', {}))
    apply_resize(img, params.get('resize', {}))
    apply_rotate(img, params.get('rotate', {}))
    apply_text(img, params.get('text', {}))
    return img


def augment_one_s(task):
    """
    Обработка изображения согласно параметрам.

    Вынесена из классов для корректной работы с потоками. Функционально копия Images.augment_one

    :param task: Изображение класса Image и параметры аугментации в формате словаря.
    :return: Обработанное изображение
    """
    try:
        img, params = task
        if img.image is not None and img.image.size > 0:
            img = apply_augmentations(img, params)
        logger.info(f'Изображение {img.path} успешно обработано')
        return img
    except Exception as e:
        logger.error(f'Ошибка аугментации изображения {img.path} : {e}')
        return None