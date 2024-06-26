import random
import time
import Augmentation
import Interface
import os


def gui_work():
    """
    Запускает графический интерфейс для настройки параметров и выполняет обработку изображений на основе введённых
    данных.

    :return: Успешность выполнения.
    """
    try:
        Interface.create_gui()
        params = Interface.on_button_clicked()
        Img_Processor = Augmentation.Images()
        Img_Processor.open_folder(params['path_in'])
        Img_Processor.augmentation_random_parallel(params)
        dir = os.path.dirname((params['path_out']))
        if not os.path.exists(dir):
            os.makedirs(dir)
        Img_Processor.save_to(params['path_out'])
        Img_Processor.clear()
        return True
    except KeyboardInterrupt:
        print('Прервано пользователем')
        return False

    except Exception as e:
        print('Ошибка в ходе выполнения:', e, "\n\nПопробуйте ещё раз\n\n")
        gui_work()



#ToDo
"""
- Логирование
- Дополнительные тесты
- Презентация + Кодварс + Модули-тесты
"""

if __name__ == "__main__":
    gui_work()
