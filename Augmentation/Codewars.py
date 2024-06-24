def Digital_root(num):
    num = int(num)
    if num < 0:
        raise Exception('Only positive integers')
    while num > 9:
        tmp = [int(x) for x in str(num)]
        num = sum(tmp)
    return num


def score(dice):
    score = 0
    counter = [0] * 7
    for i in dice:
        counter[i] += 1
    print(counter)
    points3 = [0, 1000, 200, 300, 400, 500, 600]
    for i in range(len(counter)):
        if counter[i] >= 3:
            counter[i] -= 3
            score += points3[i]
    score += counter[1] * 100 + counter[5] * 50
    return score


def next_smaller(n):
    digits = list(str(n))
    length = len(digits)

    # Ищем первое число
    for i in range(length - 2, -1, -1):
        if digits[i] > digits[i + 1]:
            break
    else:
        return -1  # Если нет такого - значит это минимальное число из этих цифр

    # Находим второе число
    for j in range(length - 1, i, -1):
        if digits[j] < digits[i]:
            break

    # Меняем числа местами
    digits[i], digits[j] = digits[j], digits[i]

    # Сортируем оставшуюся часть в порядке убывания и соединяем части
    digits = digits[:i + 1] + sorted(digits[i + 1:], reverse=True)

    # Проверяем на ведущий ноль
    result = int(''.join(digits))
    if digits[0] == '0':
        return -1

    return result


def next_bigger(n):
    digits = list(str(n))
    length = len(digits)

    # Ищем первое число
    for i in range(length - 2, -1, -1):
        if digits[i] < digits[i + 1]:
            break
    else:
        return -1  # Если нет такого - значит это минимальное число из этих цифр

    # Находим второе число
    for j in range(length - 1, i, -1):
        if digits[j] > digits[i]:
            break

    # Меняем числа местами
    digits[i], digits[j] = digits[j], digits[i]

    # Сортируем оставшуюся часть в порядке убывания и соединяем части
    digits = digits[:i + 1] + sorted(digits[i + 1:], reverse=False)

    # Проверяем на ведущий ноль
    if digits[0] == '0':
        return -1
    result = int(''.join(digits))

    return result


def minor(sqarr, i, j):
    return [row[:j] + row[j + 1:] for row in (sqarr[:i] + sqarr[i + 1:])]


def determinant(sqarr):
    if len(sqarr) == 1:  # Базовый случай
        return sqarr[0][0]
    det = 0
    for i in range(len(sqarr)):
        det += ((-1) ** i) * sqarr[0][i] * determinant(minor(sqarr, 0, i))  # Считаем по определению из задачи
    return det


class EvenOrOdd:
    def __call__(self, num):
        return 'Odd' if num % 2 else 'Even'

    def __getitem__(self, index):
        return 'Odd' if index % 2 else 'Even'


class PaginationHelper:
    def __init__(self, info, num):
        self.items = info
        self.num = num

    def page_count(self):
        return (len(self.items) + self.num - 1) // self.num

    def item_count(self):
        return len(self.items)

    def page_item_count(self, page):
        if page < 0 or page >= self.page_count():
            return -1
        if page == self.page_count() - 1:
            return len(self.items) % self.num or self.num
        return self.num

    def page_index(self, item_index):
        if item_index < 0 or item_index >= len(self.items):
            return -1
        return item_index // self.num


def rgb(r, g, b):
    return f"{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


def make_readable(seconds):
    secs = seconds % 60
    time = seconds // 60
    mins = time % 60
    hours = time // 60
    return f'{hours:02d}:{mins:02d}:{secs:02d}'


def pig_latin(sentence):
    result = []
    word = ""

    for char in sentence:
        if char.isalpha():
            word += char
        else:
            if word:
                result.append(word[1:] + word[0] + "ay")
                word = ""
            result.append(char)

    # Добавляем последнее слово, если есть
    if word:
        result.append(word[1:] + word[0] + "ay")

    return ''.join(result)


def generate_hashtag(s):
    if not s:
        return False
    res = '#'
    for char in s.title():
        if char.isalpha():
            res += char
    if len(res) > 140:
        return False
    return res


def dir_reduc(arr):
    pairs = {'NORTH': 'SOUTH', 'SOUTH': 'NORTH', 'EAST': 'WEST', 'WEST': 'EAST'}
    stack = []

    for direction in arr:
        if stack and pairs[direction] == stack[-1]:
            stack.pop()
        else:
            stack.append(direction)

    return stack


def product_fib(_prod):
    fib = [0, 1]
    while fib[-1] * fib[-2] < _prod:
        fib.append(fib[-1] + fib[-2])
    if fib[-1] * fib[-2] == _prod:
        return [fib[-2], fib[-1], True]
    return [fib[-2], fib[-1], False]


def perimeter(n):
    # ## В лоб - слишком медленно, порядка 10с
    # fib = [0, 1]
    # for i in range(n):
    #     fib.append(fib[-1] + fib[-2])
    # return 4*sum(fib)
    ## Считая сумму на лету получаем удвоение эффективности, ~5с на те же данные
    if n == 0:
        return 4
    elif n == 1:
        return 8

    a, b = 0, 1
    fib_sum = 1

    for _ in range(2, n + 2):
        a, b = b, a + b
        fib_sum += b

    return 4 * fib_sum


def sum_pairs(ints, s):
    seen = {}
    for index, number in enumerate(ints):
        target = s - number
        if target in seen:
            return [target, number]
        seen[number] = index
    return None


def validate_password(password):
    import re
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d]{6,}$'
    if re.match(pattern, password):
        return True
    return False


def ip_count(start, end):
    res = 0
    first = list(map(int, start.split('.')))
    second = list(map(int, end.split('.')))
    for i in range(len(first)):
        res += (second[i] - first[i]) * 256 ** (3 - i)
    return res


def sequre(password):
    import re
    pattern = r'^[A-Za-z\d]{1,}$'
    if re.match(pattern, password):
        return True
    return False


def int32_to_ip(int32):
    return f'{(int32 >> 24) & 0xFF}.{(int32 >> 16) & 0xFF}.{(int32 >> 8) & 0xFF}.{int32 & 0xFF}'  # Ez!


cache = [0, 1]


def fibonacci(n):
    def fib_helper(n):
        if n < len(cache):
            return cache[n]
        result = fib_helper(n - 1) + fib_helper(n - 2)
        cache.append(result)
        return result

    return fib_helper(n)


if __name__ == '__main__':
    print(fibonacci(994))
