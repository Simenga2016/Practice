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


def add(x):
    def inner_sum(y=None):
        nonlocal x
        if y is None:
            return x
        x += y
        return inner_sum

    return inner_sum


class User:
    def __init__(self):
        self.rank = -8
        self.progress = 0

    def upgrade(self):
        while self.progress >= 100 and self.rank < 8:
            self.progress -= 100
            self.rank += 1
            if self.rank == 0:
                self.rank += 1
        if self.rank == 8:
            self.progress = 0

    def inc_progress(self, rank):
        if rank > 8 or rank < -8 or rank == 0:
            raise Exception("Invalid rank")
        # Calculate the effective rank difference, considering the absence of rank 0
        effective_rank = rank
        effective_self_rank = self.rank

        if effective_rank > 0 and self.rank < 0:
            effective_rank -= 1
        elif effective_rank < 0 and self.rank > 0:
            effective_self_rank -= 1

        if effective_self_rank > effective_rank + 1:
            return  # No progress if the rank difference is more than one in the negative direction

        if effective_self_rank == effective_rank + 1:
            self.progress += 1
        elif effective_self_rank == effective_rank:
            self.progress += 3
        else:
            self.progress += 10 * (effective_rank - effective_self_rank) ** 2

        self.upgrade()

    def rank(self):
        return self.rank

    def progress(self):
        return self.progress


def count_calls(func, *args, **kwargs):
    import sys
    call_count = -1

    def trace_calls(frame, event, arg):
        nonlocal call_count
        if event == 'call':
            call_count += 1
        return trace_calls

    sys.settrace(trace_calls)
    func(*args, **kwargs)
    sys.settrace(None)

    return call_count


def beeramid(bonus, price):
    count = bonus // price
    i = 0
    while count > 0:
        count -= i ** 2
        print(count)
        i += 1
    if count < 0:
        i -= 1
    i -= 1
    return i if i >= 0 else 0


def mean(array_a, array_b):
    num = 0
    denum = 0
    for _ in range(len(array_a)):
        num += (array_a[_] - array_b[_]) ** 2
        denum += 1
    return num / denum


def pascal_to_snake_case(s):
    from re import sub
    return sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def snail(array):
    result = []
    if array == [[]]:
        return [[]]
    n = len(array)
    top, bottom = 0, n - 1
    left, right = 0, n - 1

    while top <= bottom and left <= right:
        # left to right
        for i in range(left, right + 1):
            result.append(array[top][i])
        top += 1

        # top to bot
        for i in range(top, bottom + 1):
            result.append(array[i][right])
        right -= 1

        if top <= bottom:
            # right to left
            for i in range(right, left - 1, -1):
                result.append(array[bottom][i])
            bottom -= 1

        if left <= right:
            # bot to top
            for i in range(bottom, top - 1, -1):
                result.append(array[i][left])
            left += 1

    return result


def to_roman(num):
    val = [
        1000, 900, 500, 400,
        100, 90, 50, 40,
        10, 9, 5, 4,
        1
    ]
    syb = [
        "M", "CM", "D", "CD",
        "C", "XC", "L", "XL",
        "X", "IX", "V", "IV",
        "I"
    ]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num


def from_roman(s):
    roman = {
        'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
        'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
        'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
    }
    i = 0
    num = 0
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] in roman:
            num += roman[s[i:i + 2]]
            i += 2
        else:
            num += roman[s[i]]
            i += 1
    return num


import copy
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_live_neighbors(grid, row, col):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    live_neighbors = 0

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1:
            live_neighbors += 1

    return live_neighbors


def next_generation(grid):
    rows, cols = len(grid), len(grid[0])
    new_grid = copy.deepcopy(grid)

    for row in range(rows):
        for col in range(cols):
            live_neighbors = get_live_neighbors(grid, row, col)
            if grid[row][col] == 1:  # Live cell
                if live_neighbors < 2 or live_neighbors > 3:
                    new_grid[row][col] = 0
            else:  # Dead cell
                if live_neighbors == 3:
                    new_grid[row][col] = 1

    return new_grid


def crop_grid(grid):
    rows = len(grid)
    cols = len(grid[0])
    top, bottom, left, right = rows, 0, cols, 0

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                if row < top: top = row
                if row > bottom: bottom = row
                if col < left: left = col
                if col > right: right = col

    if top > bottom or left > right:
        return [[]]

    return [row[left:right + 1] for row in grid[top:bottom + 1]]


def expand_grid(grid, padding=1):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0] * (cols + 2 * padding) for _ in range(rows + 2 * padding)]

    for row in range(rows):
        for col in range(cols):
            new_grid[row + padding][col + padding] = grid[row][col]

    return new_grid


def game_of_life(grid, generations):
    grid = expand_grid(grid, padding=generations)
    grids = [copy.deepcopy(grid)]
    for _ in range(generations):
        grid = next_generation(grid)
        grids.append(copy.deepcopy(grid))
    return grids


def animate_game_of_life(grids): # Немножко поиграл дальше задачи, подкрутив визуализацию
    fig, ax = plt.subplots()
    ax.set_axis_off()
    mat = ax.matshow(grids[0], cmap='binary')

    def update(frame):
        mat.set_data(grids[frame])
        return [mat]

    ani = animation.FuncAnimation(fig, update, frames=len(grids), interval=200, blit=True)
    plt.show()

def spin_words(sentence):
    words = sentence.split()
    spun_words = [word[::-1] if len(word) >= 5 else word for word in words]
    return ' '.join(spun_words)

if __name__ == '__main__':
    print(pascal_to_snake_case('Agg3hHfg'))
