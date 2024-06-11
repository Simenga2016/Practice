def Digital_root(num):
    num = int(num)
    if num < 0 :
        raise Exception('Only positive integers')
    while num > 9:
        tmp = [int(x) for x in str(num)]
        num = sum(tmp)
    return num

def score(dice):
    score = 0
    counter = [0] * 7
    for i in dice:
        counter[i]+=1
    print(counter)
    points3 = [0,1000,200,300,400,500,600]
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

def minor(sqarr,i,j):
    return [row[:j] + row[j+1:] for row in (sqarr[:i]+sqarr[i+1:])]

def determinant(sqarr):
    if len(sqarr) == 1: # Базовый случай
        return sqarr[0][0]
    det = 0
    for i in range(len(sqarr)):
        det += ((-1)**i) * sqarr[0][i] * determinant(minor(sqarr,0,i)) # Считаем по определению из задачи
    return det

class EvenOrOdd:
    def __call__(self, num):
        return 'Odd' if num % 2 else 'Even'

    def __getitem__(self, index):
        return 'Odd' if index % 2 else 'Even'

# Пример использования:
even_or_odd = EvenOrOdd()

if __name__ == '__main__':

    print(even_or_odd[3])


