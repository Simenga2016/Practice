# Импортируем необходимые библиотеки
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
    cv2.imwrite(output_path, image)


def rotate_image(image, angle):
    """
    Поворачивает изображение на заданный угол.

    :param image: Исходное изображение.
    :param angle: Угол поворота в градусах.
    :return: Повернутое изображение.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image


def flip_vertical(image):
    """
    Отражает изображение по вертикали.

    :param image: Исходное изображение.
    :return: Отраженное изображение.
    """
    flipped_image = cv2.flip(image, 0)
    return flipped_image


def flip_horizontal(image):
    """
    Отражает изображение по горизонтали.

    :param image: Исходное изображение.
    :return: Отраженное изображение.
    """
    flipped_image = cv2.flip(image, 1)
    return flipped_image


def resize_image(image, width=None, height=None):
    """
    Изменяет размер изображения.

    :param image: Исходное изображение.
    :param width: Новая ширина изображения.
    :param height: Новая высота изображения.
    :return: Измененное изображение.
    """
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


def crop_image(image, x, y, width = 0, height = 0):
    """
    Обрезает изображение до заданных размеров и координат.

    :param image: Исходное изображение.
    :param x: Координата x верхнего левого угла области обрезки.
    :param y: Координата y верхнего левого угла области обрезки.
    :param width: Ширина области обрезки.
    :param height: Высота области обрезки.
    :return: Обрезанное изображение.
    """
    if not width and height:
        cropped_image = image[y:y + height, x:]
    elif width and not height:
        cropped_image = image[y:, x:x + width]
    elif not width and not height:
        cropped_image = image[y:, x:]
    else:
        cropped_image = image[y:y + height, x:x + width]
    return cropped_image

def visualize_image(image):
    """
    Визуализирует изображение с помощью matplotlib.

    :param image: Изображение для визуализации.
    """
    # Конвертируем изображение из BGR (OpenCV) в RGB (PIL)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')  # Скрываем оси
    plt.show()


# Пример использования функций
input_image_path = 'test.gif'  # Замените на путь к вашему входному изображению
output_image_path = './output_image.jpg'  # Замените на путь к вашей выходной директории
rotation_angle = -22  # Угол поворота в градусах

# Создаем выходную директорию, если она не существует
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Открываем изображение
image = open_image(input_image_path)
image = resize_image(image,640,400)
image = crop_image(image,17,40)

# Поворачиваем изображение
rotated_image = rotate_image(image, rotation_angle)
fliped_image = flip_vertical(flip_horizontal(flip_vertical(flip_horizontal(rotated_image))))
# Сохраняем повернутое изображение
save_image(fliped_image, output_image_path)

# Визуализируем повернутое изображение
visualize_image(fliped_image)

# if __name__ == '__main__':
#     print(read_file('test.jpeg').read())

