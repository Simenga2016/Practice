import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import random
import os
import concurrent.futures

class Image:
    def __init__(self, input_path):
        self.image = self.open_image(input_path)

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

    def save_image(self, output_path):
        """
        Сохраняет изображение с помощью OpenCV.
        :param output_path: Путь для сохранения изображения.
        """
        try:
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

    def blur_image(self, power=(11, 11)):
        """
        Размывает изображение с использованием гауссового размытия.
        :param power: Сила размытия по осям (кортеж из двух нечетных целых чисел).
        :return: Размытое изображение.
        """
        try:
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
            height = height if height else random.randint((shape[0] - x) // 10, shape[0] - 1)
            width = width if width else random.randint((shape[1] - y) // 10, shape[1] - 1)
            x = x if x else random.randint(shape[0] // 10, shape[0] - height)
            y = y if y else random.randint(shape[1] // 10, shape[1] - width)
            return self.crop_image(x, y, width, height)
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
            if blur and not (isinstance(blur_power, tuple) and len(blur_power) == 2 and all(
                    isinstance(x, int) for x in blur_power)):
                raise ValueError("Параметр 'blur_power' должен быть кортежем из двух целых чисел.")
            if blur and not (all(x > 0 and x % 2 == 1 for x in blur_power)):
                raise ValueError("Значения в параметре 'blur_power' должны быть нечетными и больше нуля.")

            mask = np.zeros_like(self.image)
            cv2.putText(mask, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)

            if blur:
                mask = cv2.GaussianBlur(mask, blur_power, 0)
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
            raise Exception(f'Ошибка добавления шума: {e}')

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
            raise Exception(f'Ошибка изменения контраста: {e}')

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
            raise Exception(f'Ошибка изменения яркости: {e}')

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
            raise Exception(f'Ошибка изменения насыщенности: {e}')

    def visualize_image(self):
        """
        Визуализирует изображение с помощью matplotlib.
        """
        try:
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.show()
        except Exception as e:
            raise Exception(f'Ошибка визуализации: {e}')

    def process_image(self, function_tuples):
        """
        Выполняет функции обработки изображения.

        :param function_tuples: Список кортежей (функция, аргументы), которые нужно выполнить.
        """
        try:
            for func, args in function_tuples:
                func(*args)
        except Exception as e:
            print(f'Ошибка выполнения функции: {e}')

    def process_images_parallel(self, process_functions):
        """
        Выполняет функции обработки изображений параллельно.

        :param process_functions: Список списков кортежей (функция, аргументы), которые нужно выполнить.
        :return: Список обработанных изображений.
        """
        processed_images = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for functions in process_functions:
                futures.append(executor.submit(self.process_image, functions))
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Ждем завершения выполнения
                    processed_images.append(self.image.copy())  # Добавляем обработанное изображение в список
                except Exception as e:
                    print(f'Ошибка выполнения процесса: {e}')
        return processed_images

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
    image = Image(input_image_path)

    # Определяем список списков функций для каждого процесса обработки изображений
    process_functions = [
        [
            (image.resize_image, (640, 400)),
            (image.blur_image, (19, 41)),
        ],
        [
            (image.change_saturation, (4.0,)),
            (image.random_crop_image, (800, 600)),
            (image.change_contrast, (0.5,)),
        ],
    ]
    for _ in range(11):
        process_functions.append([
            [(image.random_crop_image, (320, 400)), ]
        ])

    # Выполняем функции параллельно
    processed_images = image.process_images_parallel(process_functions)

    # Визуализируем обработанные изображения
    for idx, processed_image in enumerate(processed_images):
        cv2.imshow(f'Processed Image {idx}', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

