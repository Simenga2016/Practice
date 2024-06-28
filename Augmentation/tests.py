"""
Тесты аугментатора. Включают в себя:
- Проверка изменения насыщенности
- Проверка изменения яркости
- Проверка изменения контраста
- Проверка обрезания
- Проверка изменения размеров
- Проверка изменения поворота
- Проверка изменения отражения па обеим осям
- Проверка изменения размытия
- Проверка на открытие и сохранение в различных форматах
- Проверка изменения формата
- Проверка одновременного воздействия разных аугментаций
- Проверка открытия несуществующего файла (Возвращает Warning)
- Проверка открытия файла неверного формата (.txt файл без структуры)
- Проверка открытия неподдреживаемого формата (ASCII-арт в файле .txt)


"""

import cv2
import numpy as np
import Augmentation


def test_saturation():
    """
    Проверяем открытие и изменение насыщенности.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.change_saturation(2)
    img.save_image('./test/test_tmp.jpg')

    img_test = cv2.imread('./test/test_tmp.jpg')
    expected_img = cv2.imread('./test/Expected/saturation.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Обработка насыщенности не верна!"


def test_brightness():
    """
    Проверяем открытие и изменение яркости.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.change_brightness(0.4)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/brightness.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Обработка яркости не верна!"


def test_contrast():
    """
    Проверяем открытие и изменение контраста.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.change_contrast(3 ** 0.5)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/contrast.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Обработка контраста не верна!"


def test_crop():
    """
    Проверяем открытие и обрезку.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.crop_image(4, 12, 16, 16)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/crop.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Обрезка не верна!"


def test_resize():
    """
    Проверяем открытие и изменение размера.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.resize_image(128, 128)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/resize.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Изменение размера не верно!"


def test_rotate():
    """
    Проверяем открытие и поворот.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.rotate_image(17)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/rotate17.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Вращение не верно!"


def test_flip():
    """
    Проверяем открытие и отзеркаливание.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.flip_horizontal()
    img.flip_vertical()
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/fliped.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Отражение не верно!"


def test_blur():
    """
    Проверяем открытие и размытие.

    :return: None
    """

    img = Augmentation.Image('./test/test.jpg')
    img.blur_image(9, 24)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/blur.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Размытие не верно!"


def test_bmp():
    """
    Проверяем открытие и сохранение в формате bmp.

    :return: None
    """

    img = Augmentation.Image('./test/test.bmp')
    img.save_image('./test/test_tmp.bmp')

    expected_img = cv2.imread('./test/Expected/bmp.bmp')
    img_test = cv2.imread('./test/test_tmp.bmp')

    assert np.allclose(img_test, expected_img, atol=1), "Обработка формата bmp не верна!"


def test_png():
    """
    Проверяем открытие и сохранение в формате png.

    :return: None
    """

    img = Augmentation.Image('./test/test.png')
    img.save_image('./test/test_tmp.png')

    expected_img = cv2.imread('./test/Expected/png.png')
    img_test = cv2.imread('./test/test_tmp.png')

    assert np.allclose(img_test, expected_img, atol=1), "Обработка формата png не верна!"

def test_convert():
    """
    Проверяем открытие и сохранение в формате png.

    :return: None
    """

    img = Augmentation.Image('./test/test.png')
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/convert_png.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Конвертирование не верно!"


def test_complex():
    img = Augmentation.Image('./test/test.jpg')
    img.rotate_image(5)
    img.resize_image(128, 128)
    img.blur_image(4, 8)
    img.crop_image(0, 0, 96, 96)
    img.change_contrast(1.4)
    img.change_saturation(2.1)
    img.change_brightness(0.4)
    img.flip_horizontal()
    img.resize_image(192, 192)
    img.save_image('./test/test_tmp.jpg')

    expected_img = cv2.imread('./test/Expected/complex.jpg')
    img_test = cv2.imread('./test/test_tmp.jpg')

    assert np.allclose(img_test, expected_img, atol=1), "Отражение не верно!"

def test_nothing():
    img = Augmentation.Image('.path/to/nothing/there.nothing')
    assert img == None or img.image == None, "Найдено что-то в пустоте!"

def test_not_image():
    img = Augmentation.Image('./test/cat.txt')
    assert img == None or img.image == None, "Читаем текст вместо изображения!"

def test_void():
    img = Augmentation.Image('./test/Void.txt')
    img.visualize_image()
    assert img == None or img.image == None, "Читаем пустоту!"
