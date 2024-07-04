from selenium import webdriver  # Импортируем модуль webdriver из библиотеки Selenium

# Настраиваем параметры для веб-драйвера Chrome
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Запускаем браузер в фоновом режиме (без графического интерфейса)

# Инициализируем веб-драйвер с заданными параметрами
driver = webdriver.Chrome(options=options)

# Открываем веб-страницу по указанному URL
driver.get('https://github.com/Simenga2016')

# Используем JavaScript для установки элемента в localStorage
driver.execute_script("localStorage.setItem('Foo', 'Bar');")

# Получаем значение элемента из localStorage с помощью JavaScript
value = driver.execute_script("return localStorage.getItem('Foo');")
print(f"Значение из localStorage: {value}")  # Выводим значение на консоль

# Проверяем сохранность данных по всему сайту
driver.get('https://github.com/')

value = driver.execute_script("return localStorage.getItem('Foo');")
print(f"Значение из localStorage в ином месте сайта: {value}")  # Выводим значение на консоль

# Удаляем элемент из localStorage с помощью JavaScript
driver.execute_script("localStorage.removeItem('Foo');")

# Пытаемся получить значение элемента из localStorage после его удаления
value_after_deletion = driver.execute_script("return localStorage.getItem('Foo');")
print(f"Значение после удаления: {value_after_deletion}")  # Выводим результат на консоль

# Закрываем браузер
driver.quit()
