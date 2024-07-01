import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import random
from copy import deepcopy
import os
import multiprocessing
from logger import logger
from Augment import augment_one_s
from  Random_augment import *

class Image:
    def __init__(self, input_path=None, image=None):
        self.image = self.open_image(input_path) if input_path != None else image
        self.path = input_path if input_path is not None else 'Copy'

    def __call__(self):
        """
        Возвращает изображение, хранящееся в объекте класса.
        """
        return self.image

    def open_image(self, input_path):
        """
        Открывает изображение с помощью OpenCV или PIL, в зависимости от расширения файла.

        :param input_path: Путь к входному изображению.
        :return: Открытое изображение (numpy array).
        """
        try:
            image = cv2.imread(input_path)
            if image is None:
                logger.warning(f"Не удалось открыть изображение по пути: {input_path}")
                raise Exception
            logger.info(f'File {input_path} opened successfully')
            return image
        except:
            try:
                with PILImage.open(input_path) as img:
                    img.load()
                    if img.mode == 'L':
                        image = np.array(img.convert('RGB'))
                    elif img.mode == 'P':
                        image = np.array(img.convert('RGB'))
                    else:
                        image = np.array(img)
                    res = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    logger.info(f'File {input_path} opened successfully')
                    return res
            except Exception as e:
                logger.error(f"Не удалось открыть изображение по пути: {input_path}. Ошибка: {str(e)}")

    def copy(self):
        """
        Создаёт идентичный объект класса Image.

        :return: Копия изображения.
        """
        try:
            if self.image or  self.image.size > 0:
                return deepcopy(self)
        except Exception as e:
            logger.error(f'Ошибка копирования {e}')

    def save_image(self, output_path):
        """
        Сохраняет изображение с помощью OpenCV.

        :param output_path: Путь для сохранения изображения.
        """
        try:
            if self.image is not None and self.image.size > 0:
                cv2.imwrite(output_path, self.image)
                logger.debug(f'File {output_path} successfully saved')
        except Exception as e:
            logger.error(f'Ошибка сохранения: {e} ')

    def rotate_image(self, angle):
        """
        Поворачивает изображение на заданный угол.

        :param angle: Угол поворота в градусах.
        :return: Повернутое изображение.
        """
        try:
            if self.image is not None and self.image.size >0:
                (h, w) = self.image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                self.image = cv2.warpAffine(self.image, M, (w, h))
        except Exception as e:
            logger.error(f'Ошибка поворота изображения {self.path}: {e}')

    def flip_vertical(self):
        """
        Отражает изображение по вертикали.

        :return: Отраженное изображение.
        """
        try:
            if self.image is not None and self.image.size >0:
                self.image = cv2.flip(self.image, 0)
        except Exception as e:
            logger.error(f'Ошибка отзеркаливания изображения {self.path}: {e}')

    def flip_horizontal(self):
        """
        Отражает изображение по горизонтали.

        :return: Отраженное изображение.
        """
        try:
            if self.image is not None and self.image.size >0:
                self.image = cv2.flip(self.image, 1)
        except Exception as e:
            logger.error(f'Ошибка отзеркаливания изображения {self.path}: {e}')

    def resize_image(self, width=None, height=None):
        """
        Изменяет размер изображения.

        :param width: Новая ширина изображения.
        :param height: Новая высота изображения.
        """
        try:
            if self.image is not None and self.image.size > 0:
                if not width:
                    width = None
                if not height:
                    height = None
                if width is None and height is None:
                    raise ValueError("Необходимо указать хотя бы один из параметров: width или height.")

                if width is None:
                    aspect_ratio = height / float(self.image.shape[0])
                    new_width = int(self.image.shape[1] * aspect_ratio)
                    dim = (new_width, height)
                elif height is None:
                    aspect_ratio = width / float(self.image.shape[1])
                    new_height = int(self.image.shape[0] * aspect_ratio)
                    dim = (width, new_height)
                else:
                    dim = (width, height)

                self.image = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f'Ошибка изменения размера изображения {self.path}: {e}')

    def crop_image(self, x, y, width=0, height=0):
        """
        Обрезает изображение до заданных размеров и координат.

        :param x: Координата x верхнего левого угла области обрезки.
        :param y: Координата y верхнего левого угла области обрезки.
        :param width: Ширина области обрезки.
        :param height: Высота области обрезки.
        """
        try:
            if x < 0:
                logger.warning(f'Неправильные значения параметров обрезания! Введены: x={x}, y={y}')
                x = 0
            if y < 0:
                logger.warning(f'Неправильные значения параметров обрезания! Введены: x={x}, y={y}')
                y = 0
            if width < 0:
                width = 0
            if height < 0:
                height = 0
            if not width and height:
                self.image = self.image[y:y + height, x:]
            elif width and not height:
                self.image = self.image[y:, x:x + width]
            elif not width and not height:
                self.image = self.image[y:, x:]
            else:
                self.image = self.image[y:y + height, x:x + width]
            if self. image is None or self.image.size <= 0:
                self.image = None
        except Exception as e:
            logger.error(f'Ошибка обрезки изображения {self.path}: {e}')

    def blur_image(self, power_x=11, power_y=11):
        """
        Размывает изображение с использованием гауссового размытия.

        :param power_x: Сила размытия по оси Ox.
        :param power_y: Сила размытия по оси Oy.
        :return: Размытое изображение.
        """

        try:
            if self.image is not None and self.image.size >0:
                power_y += 1 if power_y % 2 == 0 else 0  # Исправление на нечётность - необходима для метода размытия.
                power_x += 1 if power_x % 2 == 0 else 0  # Исправление на нечётность - необходима для метода размытия.
                power = (power_x, power_y)
                if not (isinstance(power, tuple) and len(power) == 2 and all(isinstance(x, int) for x in power)):
                    raise ValueError("Параметр 'power' должен быть кортежем из двух целых чисел.")
                if not (all(x > 0 for x in power)):  # Проверка на отрицательность
                    raise ValueError("Значения в параметре 'power' должны быть больше нуля.")
                self.image = cv2.GaussianBlur(self.image, power, 0)
        except Exception as e:
            logger.error(f'Ошибка размытия изображения {self.path}: {e}')

    def random_crop_image(self, width=0, height=0, x=0, y=0):
        """
        Обрезает изображение по случайному окну.

        :param height: Высота области обрезки.
        :param width: Ширина области обрезки.
        :param x: Координата x верхнего левого угла области обрезки - если не указано, выбирается случайно.
        :param y: Координата y верхнего левого угла области обрезки - если не указано, выбирается случайно.
        :return: Обрезанное изображение.
        """
        try:
            if self.image is not None and self.image.size >0:
                shape = self.image.shape[:2]
                height = height if height else random.randint((shape[0] - 1) // 10, shape[0] - (shape[0] - 1) // 10)
                width = width if width else random.randint((shape[1] - 1) // 10, shape[1] - (shape[1] - 1) // 10)
                x = x if x else random.randint(max(shape[0] - height, 50) // 10, max(shape[0] - height, 100))
                y = y if y else random.randint(max(shape[1] - width, 50) // 10, max(shape[1] - width, 100))
                self.crop_image(x, y, width, height)
        except Exception as e:
            logger.error(f'Ошибка обрезки изображения {self.path}: {e}')

    def add_text_to_image(self, text, position=(0, 0), font_scale=1, color=(255, 255, 255), thickness=2,
                          blur=False, blur_power=(11, 11), angle=0, font=cv2.FONT_HERSHEY_SIMPLEX):
        """
        Добавляет текст на изображение с возможностью размыть текст, задать угол поворота и размер шрифта.

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
            if self.image is not None and self.image.size > 0:
                mask = np.zeros_like(self.image)
                cv2.putText(mask, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

                if blur:
                    bp1 = blur_power[0]
                    bp2 = blur_power[1]
                    bp1 += 1 if blur_power[0] % 2 == 0 else 0
                    bp2 += 1 if blur_power[1] % 2 == 0 else 0
                    mask = cv2.GaussianBlur(mask, (bp1, bp2), 0)
                if angle != 0:
                    (h, w) = mask.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    mask = cv2.warpAffine(mask, M, (w, h))

                self.image = cv2.addWeighted(self.image, 1, mask, 0.5, 0)
            else:
                pass
        except Exception as e:
            logger.error(f'Ошибка добавления текста на изображение {self.path}: {e}')

    def add_noise(self, mean=25, var=64):
        """
        Добавляет шум на изображение.
        :param mean: Среднее значение гауссовского шума.
        :param var: Дисперсия гауссовского шума.
        :return: Изображение с добавленным шумом.
        """
        try:
            if self.image is not None and self.image.size >0:
                row, col, ch = self.image.shape
                sigma = var ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                noisy = self.image + gauss.reshape(row, col, ch)
                self.image = np.clip(noisy, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error(f'Ошибка добавления шума на изображение {self.path}: {e}')

    def change_contrast(self, contrast=1.0):
        """
        Изменяет контрастность изображения.
        :param contrast: Коэффициент контрастности.
        :return: Изображение с измененной контрастностью.
        """
        try:
            if self.image is not None and self.image.size >0:
                if contrast < 0:
                    raise ValueError("Контраст должен быть неотрицательным.")
                self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=0)
        except Exception as e:
            logger.error(f'Ошибка изменения контраста изображения {self.path}: {e}')

    def change_brightness(self, brightness=1.0):
        """
        Изменяет яркость изображения.
        :param brightness: Коэффициент яркости.
        :return: Изображение с измененной яркостью.
        """
        try:
            if self.image is not None and self.image.size >0:
                if brightness < 0:
                    raise ValueError("Яркость должна быть неотрицательной.")
                self.image = cv2.convertScaleAbs(self.image, alpha=brightness, beta=0)
        except Exception as e:
            logger.error(f'Ошибка изменения яркости изображения {self.path}: {e}')

    def change_saturation(self, saturation=1.0):
        """
        Изменяет насыщенность изображения.
        :param saturation: Коэффициент насыщенности.
        :return: Изображение с измененной насыщенностью.
        """
        try:
            if self.image is not None and self.image.size >0:
                if saturation < 0:
                    logger.error("Насыщенность должна быть неотрицательной.")
                hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                hsv_image = np.float32(hsv_image)
                hsv_image[:, :, 1] *= saturation
                hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
                hsv_image = np.uint8(hsv_image)
                self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.error(f'Ошибка изменения насыщенности изображения {self.path}: {e}')

    def visualize_image(self):
        """
        Визуализирует изображение с помощью matplotlib.
        """
        try:
            if self.image is not None and self.image.size > 0:
                image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                plt.imshow(image_rgb)
                plt.axis('off')
                plt.show()
                logger.info(f'Изображение {self.path} успешно показано!')
        except Exception as e:
            logger.error(f'Ошибка визуализации изображения {self.path}: {e}')

    def augment(self, functions):
        """
        Применяет набор функций к изображению для его аугментации.

        :param functions: Список кортежей (имя функции, аргументы) для применения к изображению.
        """
        for func_name, args in functions:
            try:
                # Получаем метод по его имени
                func = getattr(self, func_name)
                func(*args)
            except Exception as e:
                logger.error(f'Ошибка применения функции {func_name}: {e}')


class Images:
    """
    Класс обработчика изображений. Содержит в себе массив объектов класса Image как self.images. Поддерживает массовую
    аугментацию всех изображений.
    """

    def __init__(self):
        self.num = 0
        self.images = []
        self.path_out = '.'
        self.random_params = {
            'multiplicator': 1,
            'blur': {
                'enable': False,
                'power_x': (5, 16),
                'power_y': (5, 16)
            },
            'brightness': {
                'enable': False,
                'range': (50, 200)
            },
            'flip': {
                'enable': False,
                'flip_code': (-1, 0, 1)  ###
            },
            'saturation': {
                'enable': False,
                'range': (50, 200)
            },
            'noise': {
                'enable': False,
                'mean_range': (4, 16),
                'variance_range': (4, 32)
            },
            'contrast': {
                'enable': False,
                'range': (50, 200)
            },
            'crop': {
                'enable': False,
                'random': False,
                'left': (0, 256),
                'top': (0, 256),
                'window_width': (64, 256),
                'window_height': (64, 256)
            },
            'resize': {
                'enable': False,
                'width_range': (512, 1024),
                'height_range': (512, 1024)
            },
            'rotate': {
                'enable': False,
                'angle_range': (-10, 10)
            },
            'text': {
                'enable': False,
                'text': 'Hello, world!',
                'position_x_range': (0, 512),
                'position_y_range': (0, 512),
                'font': cv2.FONT_HERSHEY_SIMPLEX,
                'font_scale_range': (3, 30),
                'color_range': ((0, 255), (0, 255), (0, 255)),
                'thickness_range': (1, 3),  ###
                'enable_blur': False,
                'blur_range': ((3, 37), (3, 37)),  ###
                'angle_range': (-10, 10)
            }
        }

    def clear_dir(self, path):
        """
        Удаляет все файлы с расширениями .jpg, .png, .bmp и .gif в указанной папке.

        :param path: Путь к файлу или директории, из которой нужно удалить файлы.
        """
        try:
            directory = os.path.dirname(path)
            if not os.path.exists(directory):
                logger.error(f'Путь {directory} не существует.')
                return

            if not os.path.isdir(directory):
                logger.error(f'Путь {directory} не является директорией.')
                return

            for filename in os.listdir(directory):
                if filename.endswith((".jpeg",".jpg", ".png", ".bmp", ".gif")):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.debug(f'file {path} deleted')
            logger.info(f'Папка {directory} успешно очищена!')
        except Exception as e:
            logger.error(f'Ошибка во время очищения папки {directory}: {e}')

    def open_image(self, path):
        """
        Открыть изображение по пути как объект класса Image

        :param path: Путь к изображению.
        :return: Возвращает единицу в случае успеха.
        """
        try:
            self.images.append(Image(path))
            return 1
        except Exception as e:
            logger.error(f'Ошибка открытия изображения:{e}')

    def save_to(self, path=None, extension="jpeg"):
        """
        Сохраняет все изображения в папку, предварительно очистив её от изображений.

        :param path: Путь к папке.
        :param extension: Разрешение файлов.
        :return: None
        """
        try:
            self.path_out = path if path else self.path_out
            self.clear_dir(path)
            counter = 0
            for img in self.images:
                img.save_image(f'{self.path_out}{f"({counter})" if counter else ""}.{extension}')
                counter += 1
            logger.info(f'Изображения успешно сохранены в папку {path}')
        except Exception as e:
            logger.error(f'Ошибка массового сохранения {e}')

    def clear(self):
        self.images = []

    def open_folder(self, path='.'):
        """
        Открывает все изображения в папке как объекты класса Image.

        :param path: Путь к папке.
        :return: Возвращает единицу в случае успеха.
        """
        # Допустимые расширения файлов изображений
        valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')

        try:
            # Получаем список всех файлов в указанной папке
            for file_name in os.listdir(path):
                # Полный путь к файлу
                file_path = os.path.join(path, file_name)

                # Проверяем, что это файл и что у него допустимое расширение
                if os.path.isfile(file_path) and file_name.lower().endswith(valid_extensions):
                    # Открываем изображение
                    self.open_image(file_path)
            logger.info(f'Папка {path} успешно открыта')
            return 1
        except Exception as e:
            print(f"Произошла ошибка открытия папки: {e}")

    def multiple(self, multiplicator):
        """
        Добавляет в объект класса копии всех его изображений.

        :param multiplicator: Сколько копий каждого изображения будет в итоге? (2 -> Оригинал + копия)
        """
        for _ in self.images[:]:
            for __ in range(multiplicator - 1):
                self.images.append(_.copy())

    def augmentation_all(self, functions):
        """
        Применяет все модификации ко всем объектам изображений в объекте класса.

        :param functions: Массив названий функций и аргументов в формате ('resize_image', (1024, 1024)).
        :return: Массив обработанных изображений.
        """
        try:
            for _ in self.images:
                _.augment(functions)
            logger.info(f'Изображения успешно изменены!')
            return self.images
        except Exception as e:
            raise f'Ошибка в массовой обработке: {e}'

    def augmentation_random(self, params):
        """
        Случайным образом обрабатывает все изображения объекта класса согласно параметрам.

        :param params: Словарь параметров, модифицирующий random_params.
        :return: Массив обработанных изображений.
        """

        update_parameters(self.random_params, params)

        for img in self.images:
            process_blur(img, self.random_params.get('blur', {}))
            process_brightness(img, self.random_params.get('brightness', {}))
            process_flip(img, self.random_params.get('flip', {}))
            process_saturation(img, self.random_params.get('saturation', {}))
            process_noise(img, self.random_params.get('noise', {}))
            process_contrast(img, self.random_params.get('contrast', {}))
            process_crop(img, self.random_params.get('crop', {}))
            process_resize(img, self.random_params.get('resize', {}))
            process_rotate(img, self.random_params.get('rotate', {}))
            process_text(img, self.random_params.get('text', {}))

        return self.images

    def augmentation_random_parallel(self, params={}):
        """
        Случайным образом обрабатывает все изображения объекта класса согласно параметрам.

        :param params: Словарь параметров, модифицирующий random_params.
        :return: Массив обработанных изображений.
        """

        update_parameters(self.random_params, params)

        self.multiple(self.random_params.get('multiplicator', 1))

        tasks = [[img, self.random_params] for img in self.images]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            result = pool.map(augment_one_s, tasks)

        self.images = result
        return self.images


if __name__ == "__main__":
    input_image_path = 'test1.jpg'  # Замените на путь к вашему входному изображению
    output_image_path = './output_image.png'  # Замените на путь к вашей выходной директории

    # Создаем экземпляр класса Image
    img = Image(input_image_path)

    process_functions_list = [(img.rotate_image, {'angle': 45}), (img.flip_vertical, {})]

    augmented_images = img.augment(process_functions_list)

    # Визуализация результатов
    for i, augmented_img in enumerate(augmented_images):
        print(f'Image {i + 1}:')
        augmented_img.visualize_image()
