'''import matplotlib.pyplot as plt

# Координаты точек в формате [[x1, y1, z1], [x2, y2, z2], ...]
points = [[1, 5, 1], [0, 5, 2]]

# Создаем новый график
fig = plt.figure()

# Добавляем трехмерное пространство на график
ax = fig.add_subplot(111, projection='3d')

# Извлекаем координаты x, y, z для каждой точки
x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]

# Отображаем точки на трехмерном графике
ax.scatter(x, y, z)

# Устанавливаем размеры графика
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)

# Отображаем график
plt.show()'''
'''
import math
G = [[1, 2], [3, 4], [5, 6], [7, 8]]

extr_point = 0
for i in range(0,len(G)):
    if G[extr_point][1] >= G[i][1]:
        if G[extr_point][1] == G[i][1]:
            if G[extr_point][0] > G[i][0]:
                extr_point = i
        else:
            extr_point = i
new_G = [G[extr_point]]
n = len(G)
G.pop(extr_point)
while len(new_G) < n:
    min_tan = math.atan((G[0][1]-new_G[0][1])/(G[0][0]-new_G[0][0]))
    min_i = 0
    min_dist = (G[0][0] - new_G[0][0]) ** 2 + (G[0][1] - new_G[0][1]) ** 2

    for i in range(0,len(G)):
        tan = math.atan((G[i][1]-new_G[0][1])/(G[i][0]-new_G[0][0]))
        dist = (G[i][0] - new_G[0][0]) ** 2 + (G[i][1] - new_G[0][1]) ** 2
        if tan < min_tan or (tan==min_tan and dist < min_dist):
            min_tan = tan
            min_i = i
            min_dist = dist
    new_G.append(G[min_i])
    G.pop(min_i)
'''

def raster_scan_fill(polygon):
    '''
    Реализация растровой развертки с упорядоченным списком рёбер.
    :param polygon: список вершин полигона [(x0, y0), (x1, y1), ..., (xn, yn)]
    '''
    '''points = []
    # 1️⃣ Найдём границы по Y
    y_min = min(y for _, y in polygon)
    y_max = max(y for _, y in polygon)
    # 2️⃣ Создаём список рёбер (каждое ребро - пара вершин)
    edges = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if p1[1] != p2[1]:  # исключаем горизонтальные рёбра
            if p1[1] < p2[1]:
                edges.append({'x0': p1[0], 'y0': p1[1], 'x1': p2[0], 'y1': p2[1]})
            else:
                edges.append({'x0': p2[0], 'y0': p2[1], 'x1': p1[0], 'y1': p1[1]})
    # 3️⃣ Основной цикл по сканирующим строкам
    for y in range(y_min, y_max + 1):
        intersections = []
        # 1. Найти точки пересечения со сканирующими строками
        for edge in edges:
            if edge['y0'] <= y < edge['y1']:  # пересекает ли ребро сканирующую строку?
                x0, y0, x1, y1 = edge['x0'], edge['y0'], edge['x1'], edge['y1']
                # формула для нахождения x пересечения
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                intersections.append(x_int)
        # 2. Сортировка полученного списка
        intersections.sort()
        # 3. Выделение интервалов для закраски (парами)
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                x_start = int(round(intersections[i]))
                x_end = int(round(intersections[i + 1]))
                # Тут закрашиваем пиксели от x_start до x_end на строке y
                print(f'Закрашиваем от x={x_start} до x={x_end} на строке y={y}')'''

    # 1️⃣ Построить Edge Table (ET)
    edges = []
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        if p1[1] != p2[1]:  # исключаем горизонтальные рёбра
            if p1[1] < p2[1]:
                y0, x0, y1, x1 = p1[1], p1[0], p2[1], p2[0]
            else:
                y0, x0, y1, x1 = p2[1], p2[0], p1[1], p1[0]
            inv_slope = (x1 - x0) / (y1 - y0)
            edges.append({'y_min': y0, 'y_max': y1, 'x': x0, 'inv_slope': inv_slope})
    # Сортируем Edge Table по y_min
    ET = sorted(edges, key=lambda e: e['y_min'])
    # 2️⃣ Определяем диапазон y
    y_min = min(e['y_min'] for e in ET)
    y_max = max(e['y_max'] for e in ET)
    AET = []  # Активная таблица рёбер
    # 3️⃣ Основной цикл по сканирующим строкам
    for y in range(y_min, y_max):
        # Добавляем рёбра, начинающиеся на текущей строке
        while ET and ET[0]['y_min'] == y:
            AET.append(ET.pop(0))
        # Удаляем рёбра, у которых y_max == y (закончились)
        AET = [edge for edge in AET if edge['y_max'] > y]
        # Сортируем AET по текущему x
        AET.sort(key=lambda e: e['x'])
        # 4️⃣ Рисуем интервалы парами рёбер
        for i in range(0, len(AET), 2):
            if i + 1 >= len(AET):
                break  # защита от непарных рёбер
            x_start = int(round(AET[i]['x']))
            x_end = int(round(AET[i + 1]['x']))
            print(f'Закрашиваем от x={x_start} до x={x_end} на строке y={y}')
            '''filling_points = bresenham_line(x_start, y, x_end, y)
            for point in filling_points:
                point.shade = 0.5  # для визуальной заливки
            points += filling_points'''
        # 5️⃣ Обновляем x для всех рёбер в AET
        for edge in AET:
            edge['x'] += edge['inv_slope']


def bresenham_line(x0, y0, x1, y1):
    points = []
    x =x0
    y = y0
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    s1 = 1 if x0 < x1 else -1
    s2 = 1 if y0 < y1 else -1
    c=0
    if dy>dx:
        t =dx
        dx=dy
        dy=t
        c=1
    else:
        c=0
    e=2*dy-dx
    i = 1
    while i <=dx:
        points.append({'x': int(x), 'y': int(y),'shade':1})
        while e>=0:
            if c==1:
                x =x+s1
            else:
                y = y+s2
            e = e-2*dx
        if c==1:
            y = y+s2
        else:
            x=x+s1
        e=e+2*dy
        i+=1
    return points

from collections import deque

def point_in_polygon(point, polygon):
    """Проверка, лежит ли точка внутри полигона методом трассировки луча"""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i + 1) % n]
        if ((yi > y) != (yj > y)):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            if x < x_intersect:
                inside = not inside
    return inside

def seed_fill_auto(polygon, width, height):
    """Алгоритм заливки полигона с затравкой"""
    
    # 1️⃣ Вычисляем центроид полигона (среднюю точку)
    avg_x = sum(p[0] for p in polygon) / len(polygon)
    avg_y = sum(p[1] for p in polygon) / len(polygon)
    seed_point = (int(avg_x), int(avg_y))

    # 2️⃣ Проверяем, внутри ли точка (если нет, ищем ближайшую точку внутри)
    if not point_in_polygon(seed_point, polygon):
        # Ищем ближайшую точку внутри полигона
        for dy in range(1, max(width, height)):
            for dx in range(-dy, dy + 1):
                test_point = (seed_point[0] + dx, seed_point[1] + dy)
                if 0 <= test_point[0] < width and 0 <= test_point[1] < height:
                    if point_in_polygon(test_point, polygon):
                        seed_point = test_point
                        break
            else:
                continue
            break

    print(f"Начинаем заливку с точки: {seed_point}")

    # 3️⃣ Заливаем область с затравкой (используем стек для 4-связности)
    filled = set()
    stack = deque()
    stack.append(seed_point)

    points = []

    while stack:
        x, y = stack.pop()
        if (x, y) in filled or not (0 <= x < width and 0 <= y < height):
            continue
        
        if point_in_polygon((x, y), polygon):
            filled.add((x, y))
            points.append({'x': x, 'y': y, 'shade': 0.5})  # Добавляем точку в список заливки

            # Добавляем соседние точки в стек (4-связность)
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

    print(f"Всего добавлено точек: {len(points)}")
    
    return points

# Пример полигона
polygon = [[0,0], [3,3], [5,1], [9,5], [9,0]]
polygon = [[1,1], [1,10], [8,10], [8,3], [5,5]]
result = seed_fill_auto(polygon, 800, 600)
