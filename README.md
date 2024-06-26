# Augmentator

Augmentator - это Python-программа с графическим интерфейсом, предназначенная для обработки изображений. Программа
позволяет изменять изображения в указанном диапазоне с использованием различных параметров искажений, задаваемых
пользователем через слайдеры и чекбоксы. Augmentator может работать как с отдельными изображениями, так и с целыми
папками. 

## Содержание
- [Описание](#desc)
- [Особенности](#mircs)
- [Установка](#inst)
- [Тестирование](#test)
- [Использование](#use)
- [Зависимости](#reqs)
- [Лицензия](#lic)
- [Прочее](#other)
- [To do](#ToDo)
- [Команда проекта](#devs)
- [Источники](#source)

## <a id="desc">Описание</a>
Программа предоставляет следующие возможности:

- Регулирование параметров искажения изображений через слайдеры.
- Выбор необходимых изменений с помощью чекбоксов.
- Указание адресов папок для ввода и вывода через текстовые поля.
- Ввод множителя для создания нескольких уникальных искаженных копий каждого изображения.

### **Интерфейс** предусмотрен для работы в *полноэкранном* режиме

## <a id="mircs">Особенности </a>

- **Интерфейс** реализован при помощи matplotlib, по требованию ТЗ. Реализация в файле Interface.py
- **Форматы** поддерживаемые программой: jpg, jpeg, png, gif, bmp.
- **Библиотека** реализации функционального содержания проекта в файле Augmentation.py


## <a id="inst">Установка</a>

Для установки и запуска программы выполните следующие шаги:

Клонируйте репозиторий или загрузите архив с кодом.
Установите необходимые зависимости из файла requirements.txt:

```
pip install -r requirements.txt
```

После этого программа запускается через файл *main.py*:

```Console
python main.py
```

## <a id="use">Тестирование</a>

Для запуска тестирования по 15 пунктам запустите следующее, находясь в папке проекта:

```Console
pytest ./tests.py
```

После выполнения команды ожидаемый результат примет вид:

___15 passed in 0.52s___

## <a id="use">Использование</a>

- Запустите программу:

```python
python main.py
```
- В интерфейсе укажите папку с исходными изображениями и папку для сохранения обработанных изображений.
- Настройте параметры искажений с помощью слайдеров и чекбоксов.
- Введите множитель для создания уникальных копий изображений.
- Нажмите кнопку "Принять" для начала обработки.
- Интерфейс закроется, а измененные изображения будут сохранены в указанную папку под именами out.jpg, out(1).jpg и т.д.


## <a id="reqs">Зависимости</a>
Все необходимые библиотеки перечислены в файле requirements.txt.

## <a id="lic">Лицензия

[MIT](https://choosealicense.com/licenses/mit/)

## <a id="other">Прочее</a>

Файл Augmentation.py содержит в себе используемые в работе функции и классы:

#### Image:

Класс обработчика отдельного изображения. Реализует все используемые методы:

- Масштабирование
- Поворот
- Отражение
- Обрезка
- Изменение яркости, контрастности и насыщенности
- Случайные вырезки
- Наложение текста 

А так же позволяет открывать изменять изображения форматов jpg, jpeg, png, gif, bmp, а так же сохрянять файл в любой из этих форматов. 

#### Images:

Класс обработчика массива изображений. Хранит в себе объекты класса Image и позволяет параллельно их обрабатывать. Класс реализует методы обработки всех изображений согласно словарю данных, пример в файле "Dict example.txt"

***

Файл Inteface.py содержит всё, связанное с созданием интерфейса на matplotlib по техническому заданию:
 
Функция creage_gui создаёт окно, слайдеры, кнопки и окна текстового ввода. 
 
Отдельная функция convert_data_to_random_params переводит полученную из интерфейса информацию в форму словаря для обработчика изображений класса Images.

## <a id="ToDo">To do</a>

- Больше тестов
- Версии через специальные библиотеки аугментации

## <a id="devs">Команда проекта</a>

Проект разрабатывался одним человеком:

- Тразанов Никита - разработчик

## <a id="source">Источники</a>

- [Основы OpenCV](https://habr.com/ru/articles/678570/)
- [Шпаргалка по OpenCV](https://tproger.ru/translations/opencv-python-guide)
- [Создание интерфейса на matplotlib](https://jenyay.net/Matplotlib/Widgets)
- [Справка по различным терминам](https://chatgpt.com)