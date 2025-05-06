def inside_polygon(polygon, x, y):
    """Проверка, лежит ли точка (x, y) внутри полигона с использованием алгоритма радиусного пересечения"""
    n = len(polygon)
    inside = False
    px, py = polygon[0]
    for i in range(n + 1):
        sx, sy = polygon[i % n]
        if y > min(py, sy):
            if y <= max(py, sy):
                if x <= max(px, sx):
                    if py != sy:
                        xinters = (y - py) * (sx - px) / (sy - py) + px
                    if px == sx or x <= xinters:
                        inside = not inside
        px, py = sx, sy
    return inside

def flood_fill(polygon, width, height):
    # Ищем затравочный пиксель
    start_pixel = None
    for y in range(height):
        for x in range(width):
            if inside_polygon(polygon, x, y):
                start_pixel = (x, y)
                break
        if start_pixel:
            break

    if not start_pixel:
        return []  # Если нет затравочного пикселя, возвращаем пустой список

    # Стек для хранения пикселей
    stack = [start_pixel]
    filled_ranges = []

    # Направления для поиска соседей: сверху, снизу, слева, справа
    while stack:
        x, y = stack.pop()

        # Проверяем интервал слева и справа на текущей строке
        left = x
        while left > 0 and inside_polygon(polygon, left - 1, y):
            left -= 1
        right = x
        while right < width - 1 and inside_polygon(polygon, right + 1, y):
            right += 1

        # Запоминаем диапазон
        filled_ranges.append((y, left, right))

        # Проверяем соседние строки (сверху и снизу)
        if y > 0:
            for nx in range(left, right + 1):
                if inside_polygon(polygon, nx, y - 1):
                    stack.append((nx, y - 1))
        if y < height - 1:
            for nx in range(left, right + 1):
                if inside_polygon(polygon, nx, y + 1):
                    stack.append((nx, y + 1))

    return filled_ranges

# Пример использования:
polygon = [[0, 0], [100, 0], [320, 200], [0, 100]]  # Пример полигона
width = 400
height = 400

# Получаем диапазоны для заливки
ranges = flood_fill(polygon, width, height)

# Выводим результаты
for row in ranges:
    print(f"Строка {row[0]}: от {row[1]} до {row[2]}")