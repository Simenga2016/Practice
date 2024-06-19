import requests

reqs = [
    'https://httpstat.us/101',  # 101 - смена протокола. Нет на weather.
    'https://api.openweathermap.org/data/2.5/weather?lat=53.9024716&lon=27.5618225&appid=11b75a457f13426e3a7128810188efb1',
    # 200 - данные получены.
    'https://httpstat.us/305',  # 305 - Сервер запрещает прокси - нет на weather.
    'https://api.openweathermap.org/data/2.5/weather?lat=640&lon=27.5618225&appid=11b75a457f13426e3a7128810188efb1',
    # 400 Неправильный запрос.
    'https://httpstat.us/502'  # 502 - Плохое соединение - сервер weather пока жив, не стоит DDoS-ить.
]

for _ in reqs:
    r = requests.get(_)
    print(r.status_code)
