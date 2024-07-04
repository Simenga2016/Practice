from selenium import webdriver  # Импортируем модуль webdriver из библиотеки Selenium

# Настраиваем параметры для веб-драйвера Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Запускаем браузер в фоновом режиме (без графического интерфейса)

# Инициализируем веб-драйвер с заданными параметрами
driver = webdriver.Chrome(options=options)

# Открываем веб-страницу по указанному URL
driver.get('https://github.com/Simenga2016')

# Устанавливаем cookie с именем 'Foo' и значением 'Bar'
driver.add_cookie({'name': 'Foo', 'value': 'Bar'})

# Получаем значение cookie с именем 'Foo'
cookie = driver.get_cookie('Foo')
if cookie:
    print(f"Значение cookie {cookie['name']}: {cookie['value']}")  # Выводим значение на консоль
else:
    print(f"Cookie с именем '{cookie['name']}' не найдено")

# Удаляем cookie с именем 'Foo'
driver.delete_cookie('Foo')

# Пытаемся получить значение cookie после его удаления
cookie_after_deletion = driver.get_cookie('Foo')
if cookie_after_deletion:
    print(f"Значение после удаления: {cookie_after_deletion['value']}")  # Выводим значение на консоль
else:
    print("Cookie с именем 'Foo' удалено или не найдено")

# Закрываем браузер
driver.quit()
