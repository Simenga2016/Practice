import logging
from logging.handlers import TimedRotatingFileHandler
def create_logger():
    # Создаем TimedRotatingFileHandler
    file_handler = TimedRotatingFileHandler(
        "Augmentator.log",
        when="H",
        interval=8,
        backupCount=3
    )

    # Настраиваем формат и уровень логирования для file_handler
    file_handler.setLevel(logging.INFO)
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

    # Создаем логгер
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

logger = create_logger()

if __name__ == '__main__':
    logger.warning('Test warning')
    logger.error('Test error')