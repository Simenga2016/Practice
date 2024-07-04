import logging
from logging.handlers import TimedRotatingFileHandler


def create_logger():
    """
    Создание логгера.

    Создает и настраивает логгер с двумя обработчиками:
    - TimedRotatingFileHandler для записи в файл с временным хранением
    - StreamHandler для вывода логов в консоль

    Returns:
        logging.Logger: Объект логгера, настроенный для записи в файл Augmentator.log
                        и вывода в консоль.
    """
    # Создаем TimedRotatingFileHandler для файла Augmentator.log
    file_handler = TimedRotatingFileHandler(
        "Augmentator.log",
        when="H",  # Ротация каждый час
        interval=8,  # Интервал в часах
        backupCount=3  # Хранить до 3 ротаций
    )

    # Настройка формата и уровня логирования для file_handler
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Создаем обработчик вывода логов в консоль
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    # Создаем и настраиваем логгер
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = create_logger()

if __name__ == '__main__':
    logger.warning('Test warning')
    logger.error('Test error')