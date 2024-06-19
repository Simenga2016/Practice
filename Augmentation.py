import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import random
import copy
import os
import multiprocessing

import Augmentation


class Image:
    def __init__(self, input_path=None, image=None):
        self.image = self.open_image(input_path) if input_path != None else image

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
                raise FileNotFoundError(f"Не удалось открыть изображение по пути: {input_path}")
            return image
        except Exception as e:
            try:
                with PILImage.open(input_path) as img:
                    img.load()
                    if img.mode == 'L':
                        image = np.array(img.convert('RGB'))
                    elif img.mode == 'P':
                        image = np.array(img.convert('RGB'))
                    else:
                        image = np.array(img)
                    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise FileNotFoundError(f"Не удалось открыть изображение по пути: {input_path}. Ошибка: {str(e)}")

    def copy(self):
        """
        Создаёт идентичный объект класса Image.

        :return: Копия изображения.
        """
        # return Image(image = (self.image)[:])
        return copy.deepcopy(self)

    def save_image(self, output_path):
        """
        Сохраняет изображение с помощью OpenCV.
        :param output_path: Путь для сохранения изображения.
        """
        try:
            if self.image is not None and self.image.size > 0:
                cv2.imwrite(output_path, self.image)
        except Exception as e:
            raise Exception(f'Ошибка сохранения: {e}')

    def rotate_image(self, angle):
        """
        Поворачивает изображение на заданный угол.
        :param angle: Угол поворота в градусах.
        :return: Повернутое изображение.
        """
        try:
            (h, w) = self.image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            self.image = cv2.warpAffine(self.image, M, (w, h))
        except Exception as e:
            raise Exception(f'Ошибка поворота: {e}')

    def flip_vertical(self):
        """
        Отражает изображение по вертикали.
        :return: Отраженное изображение.
        """
        try:
            self.image = cv2.flip(self.image, 0)
        except Exception as e:
            raise Exception(f'Ошибка отзеркаливания: {e}')

    def flip_horizontal(self):
        """
        Отражает изображение по горизонтали.
        :return: Отраженное изображение.
        """
        try:
            self.image = cv2.flip(self.image, 1)
        except Exception as e:
            raise Exception(f'Ошибка отзеркаливания: {e}')

    def resize_image(self, width=None, height=None):
        """
        Изменяет размер изображения.

        :param width: Новая ширина изображения.
        :param height: Новая высота изображения.
        """
        try:
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
            raise Exception(f'Ошибка изменения размера изображения: {e}')

    def crop_image(self, x, y, width=0, height=0):
        """
        Обрезает изображение до заданных размеров и координат.

        :param x: Координата x верхнего левого угла области обрезки.
        :param y: Координата y верхнего левого угла области обрезки.
        :param width: Ширина области обрезки.
        :param height: Высота области обрезки.
        """
        try:
            if not width and height:
                self.image = self.image[y:y + height, x:]
            elif width and not height:
                self.image = self.image[y:, x:x + width]
            elif not width and not height:
                self.image = self.image[y:, x:]
            else:
                self.image = self.image[y:y + height, x:x + width]
        except Exception as e:
            raise Exception(f'Ошибка обрезки: {e}')

    def blur_image(self, power_x=11, power_y=11):
        """
        Размывает изображение с использованием гауссового размытия.
        :param power: Сила размытия по осям (кортеж из двух нечетных целых чисел).
        :return: Размытое изображение.
        """
        try:
            power = (power_x, power_y)
            if not (isinstance(power, tuple) and len(power) == 2 and all(isinstance(x, int) for x in power)):
                raise ValueError("Параметр 'power' должен быть кортежем из двух целых чисел.")
            if not (all(x > 0 and x % 2 == 1 for x in power)):
                raise ValueError("Значения в параметре 'power' должны быть нечетными и больше нуля.")
            self.image = cv2.GaussianBlur(self.image, power, 0)
        except Exception as e:
            raise Exception(f'Ошибка размытия: {e}')

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
            shape = self.image.shape[:2]
            height = height if height else random.randint((shape[0] - 1) // 10, shape[0] - (shape[0] - 1) // 10)
            width = width if width else random.randint((shape[1] - 1) // 10, shape[1] - (shape[1] - 1) // 10)
            x = x if x else random.randint(max(shape[0] - height, 50) // 10, max(shape[0] - height, 100))
            y = y if y else random.randint(max(shape[1] - width, 50) // 10, max(shape[1] - width, 100))
            # print(x, y, height, width, shape)
            self.crop_image(x, y, width, height)
        except Exception as e:
            raise Exception(f'Ошибка обрезки изображения: {e}')

    def add_text_to_image(self, text, position=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=1, color=(255, 255, 255), thickness=2,
                          blur=False, blur_power=(11, 11), angle=0):
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
            mask = np.zeros_like(self.image)
            cv2.putText(mask, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

            if blur:
                mask = cv2.GaussianBlur(mask, blur_power[0], blur_power[1], 0)
            if angle != 0:
                (h, w) = mask.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                mask = cv2.warpAffine(mask, M, (w, h))

            self.image = cv2.addWeighted(self.image, 1, mask, 0.5, 0)
        except Exception as e:
            raise Exception(f'Ошибка добавления текста: {e}')

    def add_noise(self, mean=25, var=64):
        """
        Добавляет шум на изображение.
        :param mean: Среднее значение гауссовского шума.
        :param var: Дисперсия гауссовского шума.
        :return: Изображение с добавленным шумом.
        """
        try:
            row, col, ch = self.image.shape
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = self.image + gauss.reshape(row, col, ch)
            self.image = np.clip(noisy, 0, 255).astype(np.uint8)
        except Exception as e:
            # raise Exception(f'Ошибка добавления шума: {e}')
            print(f'Ошибка добавления шума: {e}')

    def change_contrast(self, contrast=1.0):
        """
        Изменяет контрастность изображения.
        :param contrast: Коэффициент контрастности.
        :return: Изображение с измененной контрастностью.
        """
        try:
            if contrast < 0:
                raise ValueError("Контраст должен быть неотрицательным.")
            self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=0)
        except Exception as e:
            # raise Exception(f'Ошибка изменения контраста: {e}')
            print(f'Ошибка изменения контраста: {e}')

    def change_brightness(self, brightness=1.0):
        """
        Изменяет яркость изображения.
        :param brightness: Коэффициент яркости.
        :return: Изображение с измененной яркостью.
        """
        try:
            if brightness < 0:
                raise ValueError("Яркость должна быть неотрицательной.")
            self.image = cv2.convertScaleAbs(self.image, alpha=brightness, beta=0)
        except Exception as e:
            # raise Exception(f'Ошибка изменения яркости: {e}')
            print(f'Ошибка изменения яркости: {e}')

    def change_saturation(self, saturation=1.0):
        """
        Изменяет насыщенность изображения.
        :param saturation: Коэффициент насыщенности.
        :return: Изображение с измененной насыщенностью.
        """
        try:
            if saturation < 0:
                raise ValueError("Насыщенность должна быть неотрицательной.")
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_image = np.float32(hsv_image)
            hsv_image[:, :, 1] *= saturation
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
            hsv_image = np.uint8(hsv_image)
            self.image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        except Exception as e:
            # raise Exception(f'Ошибка изменения насыщенности: {e}')
            print(f'Ошибка изменения насыщенности: {e}')

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
        except Exception as e:
            # raise Exception(f'Ошибка визуализации: {e}')
            print(f'Ошибка визуализации: {e}')

    def augment1(self, functions):
        """
        Применяет набор функций к изображению для его аугментации.

        :param functions: Список кортежей (функция, аргументы) для применения к изображению.
        """
        for func, args in functions:
            try:
                func(*args)
            except Exception as e:
                # raise Exception(f'Ошибка применения функции {func.__name__}: {e}')
                print(f'Ошибка применения функции {func.__name__}: {e}')

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
                # raise Exception(f'Ошибка применения функции {func_name}: {e}')
                print(f'Ошибка применения функции {func_name}: {e}')

    def multy_augment(self, process_functions_list):
        """
        Применяет наборы функций к изображению параллельно на копиях объекта класса.
        Всё ещё не работает //18.06.2024 -> Переписать в отдельный общий класс нескольких изображений?
        :param process_functions_list: Список списков кортежей (функция, аргументы) для применения к изображению.
        """
        #
        # def process_copy(process_functions, input_image):
        #     img_copy = input_image.copy()
        #     img_copy.augment(process_functions)
        #     return img_copy

        res = []
        for funcs in process_functions_list:
            coppied = self.copy()  # Создаем копию оригинального изображения для каждой операции
            print(coppied)
            coppied.augment(funcs)
            res.append(coppied)

        return res


class Images:
    def __init__(self):
        self.num = 0
        self.images = []
        self.path_out = '.'
        self.random_params = {
    'blur': {
        'enable': False,
        'power_x': (5, 100),
        'power_y': (5, 100)
    },
    'brightness': {
        'enable': False,
        'range': (10, 200)
    },
    'flip': {
        'enable': False,
        'flip_code': 0
    },
    'saturation': {
        'enable': False,
        'range': (10, 200)
    },
    'noise': {
        'enable': False,
        'mean_range': (4, 32),
        'variance_range': (4, 64)
    },
    'contrast': {
        'enable': False,
        'range': (10, 200)
    },
    'crop': {
        'enable': False,
        'random': False,
        'top_left': (4, 32),
        'top_right': (4, 64),
        'bottom_left': (4, 64),
        'bottom_right': (4, 64)
    },
    'resize': {
        'enable': False,
        'width_range': (128, 1024),
        'height_range': (128, 1024)
    },
    'rotate': {
        'enable': False,
        'angle_range': (-179, 179)
    },
    'text': {
        'enable': False,
        'text': 'Hello, world!',
        'position_x_range': (0, 1024),
        'position_y_range': (0, 1024),
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale_range': (0.5,3),
        'color_range': ((0, 255), (0, 255), (2, 255)),
        'thickness_range': (1,3),
        'line_type': False,
        'blur_range': ((3, 37), (3, 37)),
        'angle_range': (-179, 179)
    }
}

    # def __call__(self):
    #     return self

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
            # raise f'Ошибка открытия изображения:{e}'
            print(f'Ошибка открытия изображения:{e}')

    def save_to(self, path=None, extension="jpeg"):
        try:
            self.path_out = path if path else self.path_out
            counter = 0
            for img in self.images:
                img.save_image(f'{self.path_out}{f"({counter})" if counter else ""}.{extension}')
                counter += 1
        except Exception as e:
            # raise  f'Ошибка массового сохранения {e}'
            print(f'Ошибка массового сохранения {e}')

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
            return 1
        except Exception as e:
            print(f"Произошла ошибка открытия папки: {e}")

    def augmentation_all(self, functions):
        try:
            for _ in self.images:
                _.augment(functions)
            return self.images
        except Exception as e:
            # raise f'Ошибка в массовой обработке: {e}'
            print(f'Ошибка в массовой обработке: {e}')


    def augmentation_random(self, params):
        if isinstance(params,dict):
            for key, value in params.items():
                if key in self.random_params:
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key in self.random_params[key]:
                                self.random_params[key][sub_key] = sub_value
                    else:
                        self.random_params[key] = value
        
        for img in self.images:
            if self.random_params.get('blur', {}).get('enable', False):
                power_x = self.random_params.get('blur', {}).get('power_x', False)
                power_y = self.random_params.get('blur', {}).get('power_y', False)
                img.blur_image(power_x,power_y)
    
            if self.random_params.get('brightness', {}).get('enable', False):
                brightness = random.randint(
                    *self.random_params.get('brightness', {}).get('range', False))/50
                img.change_brightness(brightness)
    
            if self.random_params.get('flip', {}).get('enable', False):
                if self.random_params.get('flip', {}).get('flip_code', False):
                    img.flip_horizontal()
                else:
                    img.flip_vertical()
    
            if self.random_params.get('saturation', {}).get('enable', False):
                img.change_saturation(self.random_params['saturation'])
    
            if self.random_params.get('noise', {}).get('enable', False):
                img.add_noise(self.random_params['noise'])
    
            if self.random_params.get('contrast', {}).get('enable', False):
                img.change_contrast(self.random_params['contrast'])
    
            if self.random_params.get('crop', {}).get('enable', False):
                img.crop_image(self.random_params['crop'])
                img.random_crop_image()
    
            if self.random_params.get('resize', {}).get('enable', False):
                img.resize_image(self.random_params['resize'])
    
            if self.random_params.get('rotate', {}).get('enable', False):
                img.rotate_image(self.random_params['rotate'])
    
            if self.random_params.get('text', {}).get('enable', False):
                img.add_text_to_image(self.random_params['text'])
            
# if __name__ == '__main__':
#     input_image_path = 'test1.jpg'  # Замените на путь к вашему входному изображению
#     output_image_path = './output_image.png'  # Замените на путь к вашей выходной директории
#     rotation_angle = -11  # Угол поворота в градусах
#
#     # Создаем выходную директорию, если она не существует
#     os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
#     # Создаем экземпляр класса Image и открываем изображение
#     img = Image(input_image_path)
#
#     img.visualize_image()
#
#     img.change_saturation()
#     # img.visualize_image()
#
#     img.random_crop_image(800,600)
#     img.visualize_image()
#
#     img.blur_image((19,41))
#     img.resize_image(640,420)
#     img.visualize_image()
#
#     img.add_noise(32,100)
#     img.change_contrast(0.5)
#     img.change_brightness(1.9)
#     img.visualize_image()

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
