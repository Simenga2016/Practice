# Импортируем необходимые библиотеки
import random

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def open_image(input_path):
    """
    Открывает изображение с помощью OpenCV или PIL, в зависимости от расширения файла.

    :param input_path: Путь к входному изображению.
    :return: Открытое изображение (numpy array).
    """
    try:
        # Попробуем открыть изображение с помощью OpenCV. Этот код работал для всего, кроме gif.
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось открыть изображение по пути: {input_path}")
        return image
    except Exception as e:
        # Если не удалось открыть с OpenCV, попробуем открыть с помощью PIL. Это для gif.
        try:
            with Image.open(input_path) as img:
                img.load()  # убеждаемся, что изображение полностью загружено
                if img.mode == 'L':  # градации серого (один канал)
                    image = np.array(img.convert('RGB'))  # конвертируем в RGB
                elif img.mode == 'P':  # режим палитры (обычно для GIF)
                    image = np.array(img.convert('RGB'))  # конвертируем в RGB
                else:
                    image = np.array(img)
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Не удалось открыть изображение по пути: {input_path}. Ошибка: {str(e)}")


def save_image(image, output_path):
    """
    Сохраняет изображение с помощью OpenCV.

    :param image: Изображение для сохранения.
    :param output_path: Путь для сохранения изображения.
    """
    try:
        cv2.imwrite(output_path, image)
    except Exception as e:
        raise f'Ошибка сохранения: {e}'


def rotate_image(image, angle):
    """
    Поворачивает изображение на заданный угол.

    :param image: Исходное изображение.
    :param angle: Угол поворота в градусах.
    :return: Повернутое изображение.
    """
    try:
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image
    except Exception as e:
        raise f'Ошибка поворота: {e}'


def flip_vertical(image):
    """
    Отражает изображение по вертикали.

    :param image: Исходное изображение.
    :return: Отраженное изображение.
    """
    try:
        flipped_image = cv2.flip(image, 0)
        return flipped_image
    except Exception as e:
        raise f'Ошибка отзеркаливания: {e}'


def flip_horizontal(image):
    """
    Отражает изображение по горизонтали.

    :param image: Исходное изображение.
    :return: Отраженное изображение.
    """
    try:
        flipped_image = cv2.flip(image, 1)
        return flipped_image
    except Exception as e:
        raise f'Ошибка отзеркаливания: {e}'


def resize_image(image, width=None, height=None):
    """
    Изменяет размер изображения.

    :param image: Исходное изображение.
    :param width: Новая ширина изображения.
    :param height: Новая высота изображения.
    :return: Измененное изображение.
    """
    try:
        if width is None and height is None:
            raise ValueError("Необходимо указать хотя бы один из параметров: width или height.")

        if width is None:
            aspect_ratio = height / float(image.shape[0])
            new_width = int(image.shape[1] * aspect_ratio)
            dim = (new_width, height)
        elif height is None:
            aspect_ratio = width / float(image.shape[1])
            new_height = int(image.shape[0] * aspect_ratio)
            dim = (width, new_height)
        else:
            dim = (width, height)

        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image
    except Exception as e:
        raise f'Ошибка масштабирования: {e}'


def crop_image(image, x, y, width=0, height=0):
    """
    Обрезает изображение до заданных размеров и координат.

    :param image: Исходное изображение.
    :param x: Координата x верхнего левого угла области обрезки.
    :param y: Координата y верхнего левого угла области обрезки.
    :param width: Ширина области обрезки.
    :param height: Высота области обрезки.
    :return: Обрезанное изображение.
    """
    try:
        if not width and height:
            cropped_image = image[y:y + height, x:]
        elif width and not height:
            cropped_image = image[y:, x:x + width]
        elif not width and not height:
            cropped_image = image[y:, x:]
        else:
            cropped_image = image[y:y + height, x:x + width]
        return cropped_image
    except Exception as e:
        raise f'Ошибка обрезки: {e}'


def random_crop_image(image, height=0, width=0, x=0, y=0):
    """
    Обрезает изображение по случайному окну. Можно задать любые параметры по необходимости. Параметры, кроме image
    опциональны для задачи, если не указать - задаются в промежутке 10%-100% от исходного.

    :param image: Исходное изображение.
    :param x: Координата x верхнего левого угла области обрезки - если не указано, выбирается случайно.
    :param y: Координата y верхнего левого угла области обрезки - если не указано, выбирается случайно.
    :param width: Ширина области обрезки - если не указано, выбирается случайно.
    :param height: Высота области обрезки - если не указано, выбирается случайно.
    :return: Обрезанное изображение.
    """
    try:
        shape = image.shape[:2]
        x = x if x else random.randint(shape[0] // 10, shape[0] - 1)
        y = y if y else random.randint(shape[1] // 10, shape[1] - 1)
        height = height if height else random.randint((shape[0] - x) // 10, shape[0] - x)
        width = width if width else random.randint((shape[1] - y) // 10, shape[1] - y)
        return crop_image(image, height, width, x, y)
    except Exception as e:
        raise f'Ошибка обрезки: {e}'


def blur_image(image, power=(11, 11)):
    """
    Размывает изображение с использованием гауссового размытия.

    :param image: Исходное изображение.
    :param power: Сила размытия по осям (кортеж из двух нечетных целых чисел).
    :return: Размытое изображение.
    """
    try:
        # Проверка, что изображение не пустое и имеет правильный формат
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Изображение должно быть корректным numpy array.")

        # Проверка, что power является кортежем из двух целых чисел
        if not (isinstance(power, tuple) and len(power) == 2 and all(isinstance(x, int) for x in power)):
            raise ValueError("Параметр 'power' должен быть кортежем из двух целых чисел.")

        # Проверка, что оба значения в power нечетные и больше нуля
        if not (all(x > 0 and x % 2 == 1 for x in power)):
            raise ValueError("Значения в параметре 'power' должны быть нечетными и больше нуля.")

        # Применение гауссового размытия
        return cv2.GaussianBlur(image, power, 0)
    except Exception as e:
        raise f'Ошибка размытия: {e}'


def add_text_to_image(image, text, position=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=1, color=(255, 255, 255), thickness=2,
                      blur=False, blur_power=(11, 11),
                      angle=0):
    """
    Добавляет текст на изображение с возможностью размыть текст, задать угол поворота и размер шрифта.

    :param image: Исходное изображение.
    :param text: Текст для добавления.
    :param position: Позиция текста (кортеж (x, y)).
    :param font: Шрифт текста.
    :param font_scale: Размер шрифта.
    :param color: Цвет текста (BGR).
    :param thickness: Толщина линий текста.
    :param blur: Флаг, указывающий, нужно ли размывать текст.
    :param blur_power: Сила размытия текста (кортеж из двух нечетных целых чисел).
    :param angle: Угол поворота текста.
    :return: Изображение с добавленным текстом.
    """
    try:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Изображение должно быть корректным numpy array.")

        if blur and not (
                isinstance(blur_power, tuple) and len(blur_power) == 2 and all(isinstance(x, int) for x in blur_power)):
            raise ValueError("Параметр 'blur_power' должен быть кортежем из двух целых чисел.")

        if blur and not (all(x > 0 and x % 2 == 1 for x in blur_power)):
            raise ValueError("Значения в параметре 'blur_power' должны быть нечетными и больше нуля.")

        # Создаем маску для текста
        mask = np.zeros_like(image)

        # Рисуем текст на маске
        cv2.putText(mask, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        # Размытие текста, если необходимо
        if blur:
            mask = blur_image(mask, blur_power)

        # Поворачиваем маску, если необходимо
        if angle != 0:
            mask = rotate_image(mask, angle)

        # Накладываем маску на изображение
        image_with_text = cv2.addWeighted(image, 1, mask, 0.5, 0)

        return image_with_text
    except Exception as e:
        raise f'Ошибка добавления текста: {e}'


def add_noise(image, mean=25, var=64):
    """
    Добавляет шум на изображение.

    :param image: Исходное изображение.
    :param mean: Среднее значение гауссовского шума.
    :param var: Дисперсия гауссовского шума.
    :return: Изображение с добавленным шумом.
    """
    row, col, ch = image.shape
    mean = mean
    var = var
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss.reshape(row, col, ch)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def change_contrast(image, contrast=1.0):
    """
    Изменяет контрастность изображения.

    :param image: Исходное изображение.
    :param contrast: Коэффициент контрастности. 1.0 - без изменений, меньше 1.0 - уменьшение контраста,
                     больше 1.0 - увеличение контраста.
    :return: Изображение с измененной контрастностью.
    """
    # Убедимся, что контраст не отрицательный
    if contrast < 0:
        raise ValueError("Контраст должен быть неотрицательным.")

    # Применение коэффициента контраста
    new_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

    return new_image


def change_brightness(image, brightness=1.0):
    """
    Изменяет яркость изображения.

    :param image: Исходное изображение.
    :param brightness: Коэффициент яркости. 1.0 - без изменений, меньше 1.0 - уменьшение яркости, больше 1.0 - увеличение яркости.
    :return: Изображение с измененной яркостью.
    """
    if brightness < 0:
        raise ValueError("Яркость должна быть неотрицательной.")

    # Применение коэффициента яркости
    new_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

    return new_image


def change_saturation(image, saturation=1.0):
    """
    Изменяет насыщенность изображения.

    :param image: Исходное изображение.
    :param saturation: Коэффициент насыщенности. 1.0 - без изменений, меньше 1.0 - уменьшение насыщенности, больше 1.0 - увеличение насыщенности.
    :return: Изображение с измененной насыщенностью.
    """
    if saturation < 0:
        raise ValueError("Насыщенность должна быть неотрицательной.")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = np.float32(hsv_image)
    hsv_image[:, :, 1] *= saturation
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    hsv_image = np.uint8(hsv_image)
    new_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return new_image


def visualize_image(image):
    """
    Визуализирует изображение с помощью matplotlib.

    :param image: Изображение для визуализации.
    """
    ### Версия через чистый cv2 //- мне не нравится как выглядит, переделать
    # # Отображаем изображение в окне
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)  # Ожидаем нажатия любой клавиши
    # cv2.destroyAllWindows()  # Закрываем все окна
    ### Через matplotlib
    # Конвертируем изображение из BGR (OpenCV) в RGB (PIL)
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.axis('off')  # Скрываем оси
        plt.show()
    except Exception as e:
        raise f'Ошибка визуализации: {e}'


# Пример использования функций
input_image_path = 'test1.jpg'  # Замените на путь к вашему входному изображению
output_image_path = './output_image.png'  # Замените на путь к вашей выходной директории
rotation_angle = -11  # Угол поворота в градусах

# Создаем выходную директорию, если она не существует
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Открываем изображение
image = open_image(input_image_path)
image = random_crop_image(image, x=512, y=512)
image = resize_image(image, 640, 400)
image = crop_image(image, 17, 40)
image = blur_image(image, (19, 7))

# image = add_noise(image,6,4)
image = change_contrast(image, 0.19)
image = change_brightness(image, 7)
image = change_saturation(image, 4)
image = add_text_to_image(image, 'Hello!', (100, 200), angle=-27, blur_power=(29, 17), blur=True, font_scale=4,
                          color=(137, 0, 255), font=cv2.FONT_HERSHEY_TRIPLEX)

# Поворачиваем изображение
rotated_image = add_noise(rotate_image(image, rotation_angle))
fliped_image = flip_vertical(flip_horizontal(flip_vertical(flip_horizontal(rotated_image))))
# Сохраняем повернутое изображение
save_image(fliped_image, output_image_path)

# Визуализируем повернутое изображение
visualize_image(fliped_image)

# if __name__ == '__main__':
#     print(read_file('test.jpeg').read())
