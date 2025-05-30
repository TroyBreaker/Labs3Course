from collections.abc import Mapping
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from collections import deque

import math

app = Flask(__name__)
socketio = SocketIO(app)

def cda_line(x0, y0, x1, y1):
    points = []
    length = max(abs(x1-x0),abs(y1-y0))
    dx = (x1-x0)/length
    dy = (y1-y0)/length
    x = x0 + 0.5 * (-1 if dx < 0 else 1)
    y = y0 + 0.5 * (-1 if dy < 0 else 1)
    i = 0

    while i<=length:
        x = x + dx
        y = y + dy
        points.append({'x': int(x), 'y': int(y),'shade':1})
        i = i + 1

    return points

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

def wu_line(x0, y0, x1, y1):
    points = []
    def swap(a, b):
        return b, a

    steep = abs(y1 - y0) > abs(x1 - x0)
    if not steep:
        if x0 > x1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)

        points.append({'x': x0, 'y': y0,'shade':1})

        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx
        y = y0 + gradient

        for x in range(x0 + 1, x1):
            points.append({'x': x, 'y': y,'shade':1 - (y - int(y))})
            points.append({'x': x, 'y': y+1,'shade':y - int(y)})
            y += gradient
        points.append({'x': x1, 'y': y1,'shade':1})
    else:
        if y0 > y1:
            x0, x1 = swap(x0, x1)
            y0, y1 = swap(y0, y1)

        points.append({'x': x0, 'y': y0,'shade':1})
      
        dx = x1 - x0
        dy = y1 - y0
        gradient = dx / dy
        x = x0 + gradient

        for y in range(y0 + 1, y1):
            points.append({'x': x, 'y': y,'shade':1 - (x - int(x))})
            points.append({'x': x+1, 'y': y,'shade':x - int(x)})
            x += gradient
        points.append({'x': x1, 'y': y1,'shade':1})

    return points

def ellipse_curve(x0, y0, a, b):
    points = []
    x = 0
    y = b
    delta = a*a+b*b-2*a*a*b
    
    while y >= 0:
        x1 = x0 + x
        y1 = y0 + y
        points.append({'x': int(x1), 'y': int(y1),'shade':1})
        
        x1 = x0 + x
        y1 = y0 - y
        points.append({'x': int(x1), 'y': int(y1),'shade':1})
        
        x1 = x0 - x
        y1 = y0 - y
        points.append({'x': int(x1), 'y': int(y1),'shade':1})
        
        x1 = x0 - x
        y1 = y0 + y
        points.append({'x': int(x1), 'y': int(y1),'shade':1})
        
        if delta < 0 :
            Delta = 2*(delta + a*a*y)-1
            if Delta <=0:
                x = x+1
                delta=delta+b*b*(2*x+1)
                continue
        
        if delta > 0 :
            Delta=2*(delta - b*b*x)-1
            if Delta>0:
                y = y-1
                delta = delta+a*a*(1-2*y)
                continue
            
        x += 1
        y = y-1
        delta += b*b*(2*x+1)+a*a*(1-2*y)
    return points

def parabola_curve(x0, y0, p, maxY):
    points = []
    x = 0
    y = 0
    Sd = (1/abs(p)) * (x + 1)*(x + 1) - (y + 1)
    Sv = (1/abs(p))* x*x - (y +1)
    Sh = (1/abs(p)) * (x + 1)*(x + 1) - y

    points.append({'x': int(x0 + x), 'y': int(y0 + y),'shade':1})
    
    while y < maxY:
        if abs(Sh) <= abs(Sv):
            if abs(Sd) < abs(Sh):
                y += 1
            x += 1
        else:
            if abs(Sd) < abs(Sv):
                x += 1
            y += 1
        if p>0:
            points.append({'x': int(x0 + x), 'y': int(y0 - y),'shade':1})
            points.append({'x': int(x0 - x), 'y': int(y0 - y),'shade':1})
        if p<0:
            points.append({'x': int(x0 + x), 'y': int(y0 + y),'shade':1})
            points.append({'x': int(x0 - x), 'y': int(y0 + y),'shade':1})
        Sd = (1/abs(p)) * (x + 1)*(x + 1) - (y + 1)
        Sv = (1/abs(p))* x*x - (y +1)
        Sh = (1/abs(p)) * (x + 1)*(x + 1) - y

    return points

def hyperbola_curve(x0, y0, a, b, maxX):
    points = []
    x0 -= abs(a)
    x = abs(a)
    y = 0
    Sd = 1-(x+1)**2/a**2+(y+1)**2/b**2
    Sv = 1-x**2/a**2+(y+1)**2/b**2
    Sh = 1-(x+1)**2/a**2+y**2/b**2

    points.append({'x': int(x0 + x), 'y': int(y0 + y),'shade':1})
    
    while x < maxX + a:        
        if abs(Sh) - abs(Sv) <= 0:
            if abs(Sd) - abs(Sh) < 0:
                y += 1
            x += 1
        else:
            if abs(Sv) - abs(Sd) > 0:
                x += 1
            y += 1
        points.append({'x': int(x0 + x), 'y': int(y0 + y),'shade':1})
        points.append({'x': int(x0 + x), 'y': int(y0 - y),'shade':1})
        Sd = 1-(x+1)**2/a**2+(y+1)**2/b**2
        Sv = 1-x**2/a**2+(y+1)**2/b**2
        Sh = 1-(x+1)**2/a**2+y**2/b**2

    return points

def transpose(matrix=[[]]):
        rows = len(matrix)
        cols = len(matrix[0])
        matrix_transposed = [[0 for _ in range(rows)] for _ in range(cols)]
        for i in range(rows):
            for j in range(len(matrix[0])):
                matrix_transposed[j][i] = matrix[i][j]
        return matrix_transposed

def multiply_matrices(m1,m2):
    m2 = transpose(m2)
    rows_m1 = len(m1)
    cols_m1 = len(m1[0])
    rows_m2 = len(m2)
    cols_m2 = len(m2[0])
    result = [[0 for _ in range(rows_m2)] for _ in range(rows_m1)]
    for i in range(rows_m1):
        for j in range(rows_m2):
            for k in range(cols_m1):
                result[i][j] += m1[i][k] * m2[j][k]
    return result

def calculate_points_between(C):
    main_points = []
    points = []
    for t in range(0,31,1):
        t/=30.0
        T = [[t**3,t**2,t,1]]
        main_points.append(multiply_matrices(T,C))
    
    for i in range(0,len(main_points)-1):
        points += bresenham_line(main_points[i][0][0], main_points[i][0][1], 
                                 main_points[i+1][0][0], main_points[i+1][0][1])

    return points

def ermit_interpolation(Gn):
    Mn = [[2,-2,1,1],[-3,3,-2,-1],[0,0,1,0],[1,0,0,0]]
    for i in range(0,len(Gn[0]),1):
        Gn[2][i]-=Gn[0][i]
        Gn[3][i]-=Gn[1][i]
        Gn[2][i] *= 4
        Gn[3][i] *= 4
    C = multiply_matrices(Mn,Gn)
    return calculate_points_between(C)

def beze_interpolation(Gb):
    Mb = [[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]]
    C = multiply_matrices(Mb,Gb)
    return calculate_points_between(C)

def bspline_interpolation(G):
    Ms = [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]
    for i in range(len(Ms)):
        for j in range(len(Ms[i])):
            Ms[i][j] *= 1/6

    points = []

    for i in range(0,len(G)-3):
        Cs=[G[i],G[i+1],G[i+2],G[i+3]]
        C = multiply_matrices(Ms,Cs)
        points += calculate_points_between(C)
    return points

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

def flood_fill(polygon, width, height):
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

    #  print(f"Всего добавлено точек: {len(points)}")
    
    for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            points += bresenham_line(a[0], a[1], b[0], b[1])

    return points

def flood_fill_line(polygon, a, b):
    points = []
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
            filling_points = bresenham_line(x_start, y, x_end, y)
            for point in filling_points:
                point['shade'] = 0.5  # для визуальной заливки
            points += filling_points
        # 5️⃣ Обновляем x для всех рёбер в AET
        for edge in AET:
            edge['x'] += edge['inv_slope']
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        points += bresenham_line(a[0], a[1], b[0], b[1])

    return points

def scanline_aet(polygon):
    points = []
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
            filling_points = bresenham_line(x_start, y, x_end, y)
            for point in filling_points:
                point['shade'] = 0.5  # для визуальной заливки
            points += filling_points
            
        # 5️⃣ Обновляем x для всех рёбер в AET
        for edge in AET:
            edge['x'] += edge['inv_slope']
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        points += bresenham_line(a[0], a[1], b[0], b[1])

    return points

def scanline(polygon):
    points = []
    filling_edges = []

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
                filling_edges.append([[x_start, y],[x_end, y]])

    for edge in filling_edges:
        filling_points = bresenham_line(*edge[0], *edge[1])
        for point in filling_points:
            point['shade'] = 0.5
        points+=filling_points

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        points += bresenham_line(a[0], a[1], b[0], b[1])

    return points


import numpy as np
from scipy.spatial import Delaunay, Voronoi

def delaunay_triangulation(points):
    return_points = []
    points_np = np.array(points)
    tri = Delaunay(points_np)
    # Преобразуем каждый треугольник из numpy.array в обычный список
    triangles_coords = [points_np[triangle].tolist() for triangle in tri.simplices]
    for tri in triangles_coords:
        for i in range(3):
            x0, y0 = tri[i]
            x1, y1 = tri[(i+1)%3]
            return_points += bresenham_line(x0, y0, x1, y1)
    return return_points

def line_intersection(p0, p1, p2, p3):
    """
    Пересечение отрезков p0p1 и p2p3.
    Возвращает точку пересечения или None.
    """
    s10 = p1 - p0
    s32 = p3 - p2

    denom = s10[0]*s32[1] - s32[0]*s10[1]
    if denom == 0:
        return None  # Параллельны

    denom_is_positive = denom > 0

    s02 = p0 - p2
    s_numer = s10[0]*s02[1] - s10[1]*s02[0]
    t_numer = s32[0]*s02[1] - s32[1]*s02[0]

    if (s_numer < 0) == denom_is_positive:
        return None  # Нет пересечения
    if (t_numer < 0) == denom_is_positive:
        return None
    if (s_numer > denom) == denom_is_positive:
        return None
    if (t_numer > denom) == denom_is_positive:
        return None

    t = t_numer / denom
    intersection = p0 + t * s10
    return intersection

def clip_ray_to_rectangle(p, direction, xmin, xmax, ymin, ymax):
    """
    Обрезать луч (p + t*direction, t>=0) по прямоугольнику.
    Возвращает точку пересечения с границей прямоугольника.
    """
    intersections = []

    rect_lines = [
        (np.array([xmin, ymin]), np.array([xmax, ymin])),  # низ
        (np.array([xmax, ymin]), np.array([xmax, ymax])),  # право
        (np.array([xmax, ymax]), np.array([xmin, ymax])),  # верх
        (np.array([xmin, ymax]), np.array([xmin, ymin])),  # лево
    ]

    for p0, p1 in rect_lines:
        inter = line_intersection(p, p + direction*1e6, p0, p1)
        if inter is not None:
            # Проверяем, что точка лежит в направлении луча (t>=0)
            t = np.dot(inter - p, direction)
            if t >= 0:
                intersections.append(inter)

    if not intersections:
        return None

    # Выбираем ближайшую точку пересечения
    distances = [np.linalg.norm(inter - p) for inter in intersections]
    min_index = np.argmin(distances)
    return intersections[min_index]

def voronoi_segments_in_rectangle(points, a, b):
    try:
        points = np.array(points)
        vor = Voronoi(points)
        segments = []
        return_points = []

        xmin, xmax = 0, a
        ymin, ymax = 0, b

        for (p1_idx, p2_idx), (v1_idx, v2_idx) in zip(vor.ridge_points, vor.ridge_vertices):
            if v1_idx >= 0 and v2_idx >= 0:
                # Конечное ребро
                p1 = vor.vertices[v1_idx]
                p2 = vor.vertices[v2_idx]

                # Обрезаем отрезок по прямоугольнику (если нужно)
                # Проверим, что хотя бы часть отрезка внутри прямоугольника
                if (xmin <= p1[0] <= xmax and ymin <= p1[1] <= ymax) or \
                   (xmin <= p2[0] <= xmax and ymin <= p2[1] <= ymax):
                    # Можно просто добавить отрезок, т.к. он внутри или пересекает
                    segments.append((tuple(p1), tuple(p2)))
                else:
                    # Можно попытаться обрезать, но обычно такие ребра не нужны
                    pass

            else:
                # Бесконечное ребро
                # Найдём конечную вершину
                if v1_idx == -1:
                    v_finite = v2_idx
                else:
                    v_finite = v1_idx

                finite_vertex = vor.vertices[v_finite]

                # Точки, породившие ребро
                point1 = vor.points[p1_idx]
                point2 = vor.points[p2_idx]

                # Вектор между точками
                dp = point2 - point1
                # Нормаль к ребру (перпендикуляр)
                n = np.array([-dp[1], dp[0]])
                n /= np.linalg.norm(n)

                # Направление луча должно быть в сторону, где находится бесконечная вершина
                # Проверим направление: если вектор от finite_vertex в сторону точки n ближе к центру
                midpoint = (point1 + point2) / 2
                direction = n

                # Проверим, что направление в сторону бесконечности
                # Для этого проверим, что точка finite_vertex + direction далеко от центра точек
                center = points.mean(axis=0)
                if np.dot(finite_vertex - center, direction) < 0:
                    direction = -direction

                # Найдём пересечение луча с прямоугольником
                intersection = clip_ray_to_rectangle(finite_vertex, direction, xmin, xmax, ymin, ymax)
                if intersection is not None:
                    segments.append((tuple(finite_vertex), tuple(intersection)))

        for seg in segments:
            x0, y0 = seg[0]
            x1, y1 = seg[1]
            return_points += bresenham_line(x0, y0, x1, y1)
    except Exception as e:
        print('Ошибка:',e)

    return return_points


INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

def compute_out_code(x, y, xmin, ymin, xmax, ymax):
    code = INSIDE
    if x < xmin:
        code |= LEFT
    elif x > xmax:
        code |= RIGHT
    if y < ymin:
        code |= BOTTOM
    elif y > ymax:
        code |= TOP
    return code

def cohen_sutherland_clip(p1, p2, segments):
    # Автоматическое определение границ прямоугольника
    xmin = min(p1[0], p2[0])
    xmax = max(p1[0], p2[0])
    ymin = min(p1[1], p2[1])
    ymax = max(p1[1], p2[1])

    clipped_segments = []
    return_points = []

    for segment in segments:
        x1, y1 = segment[0]
        x2, y2 = segment[1]

        out_code1 = compute_out_code(x1, y1, xmin, ymin, xmax, ymax)
        out_code2 = compute_out_code(x2, y2, xmin, ymin, xmax, ymax)

        accept = False

        while True:
            if not (out_code1 | out_code2):
                accept = True
                break
            elif out_code1 & out_code2:
                break
            else:
                out_code_out = out_code1 if out_code1 else out_code2

                if out_code_out & TOP:
                    x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                    y = ymax
                elif out_code_out & BOTTOM:
                    x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                    y = ymin
                elif out_code_out & RIGHT:
                    y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                    x = xmax
                elif out_code_out & LEFT:
                    y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                    x = xmin

                if out_code_out == out_code1:
                    x1, y1 = x, y
                    out_code1 = compute_out_code(x1, y1, xmin, ymin, xmax, ymax)
                else:
                    x2, y2 = x, y
                    out_code2 = compute_out_code(x2, y2, xmin, ymin, xmax, ymax)

        if accept:
            clipped_segments.append([[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]])

    for seg in clipped_segments:
        x0, y0 = seg[0]
        x1, y1 = seg[1]
        return_points += bresenham_line(x0, y0, x1, y1)

    return return_points, clipped_segments

def get_drawn_edges_coords(rotation_angles, show_hidden_faces):
    """
    rotation_angles: (rx, ry, rz) в радианах
    show_hidden_faces: bool - показывать все грани или скрывать невидимые

    Возвращает список ребер [(p1, p2), ...], где p1 и p2 - 2D координаты (x, y)
    """
    rx, ry, rz = rotation_angles
    size = 100 / 2

    return_points = []

    def rotate_point(p):
        x, y, z = p

        # Вращение по X
        cosx, sinx = math.cos(rx), math.sin(rx)
        y, z = y * cosx - z * sinx, y * sinx + z * cosx

        # Вращение по Y
        cosy, siny = math.cos(ry), math.sin(ry)
        x, z = x * cosy + z * siny, -x * siny + z * cosy

        # Вращение по Z
        cosz, sinz = math.cos(rz), math.sin(rz)
        x, y = x * cosz - y * sinz, x * sinz + y * cosz

        return [x, y, z]

    vertices = [
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ]

    rotated = [rotate_point(v) for v in vertices]
    projected = [[400 + x, 300 - y, z] for x, y, z in rotated]

    faces = [
        {"vertices": [0, 1, 2, 3], "normal": [0, 0, -1]},
        {"vertices": [4, 5, 6, 7], "normal": [0, 0, 1]},
        {"vertices": [1, 5, 6, 2], "normal": [1, 0, 0]},
        {"vertices": [0, 4, 7, 3], "normal": [-1, 0, 0]},
        {"vertices": [3, 2, 6, 7], "normal": [0, 1, 0]},
        {"vertices": [0, 1, 5, 4], "normal": [0, -1, 0]}
    ]

    # Вычисляем среднюю глубину для сортировки
    for face in faces:
        z_sum = sum(projected[i][2] for i in face["vertices"])
        face["avg_z"] = z_sum / len(face["vertices"])

    faces_sorted = sorted(faces, key=lambda f: f["avg_z"], reverse=True)

    drawn_edges = set()
    edges_coords = []

    for face in reversed(faces_sorted):
        # Вращаем нормаль
        n = rotate_point(face["normal"])
        dot = n[2] * -1
        visible = dot > 0

        if show_hidden_faces or visible:
            verts = face["vertices"]
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge_key = tuple(sorted((v1, v2)))
                if edge_key not in drawn_edges:
                    drawn_edges.add(edge_key)
                    p1 = projected[v1][:2]
                    p2 = projected[v2][:2]
                    edges_coords.append((p1, p2))

    for seg in edges_coords:
        x0, y0 = seg[0]
        x1, y1 = seg[1]
        return_points += bresenham_line(x0, y0, x1, y1)

    return return_points


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('generate_line')
def handle_draw_line(data):
    points = []
    method = data['method']
    x0 = data['x0']
    y0 = data['y0']
    x1 = data['x1']
    y1 = data['y1']
    if method == 'cda':
        points = cda_line(x0, y0, x1, y1)
    elif method == 'bresenham':
        points = bresenham_line(x0, y0, x1, y1)
    elif method == "wu":
        points = wu_line(x0, y0, x1, y1)
    emit('draw', {'points': points})

@socketio.on('generate_curve')
def handle_draw_curve(data):
    points = []
    method = data['method']
    if method == 'ellips':
        x = data['x']
        y = data['y']
        a = data['a']
        b = data['b']
        points = ellipse_curve(x,y,a,b)
    elif method == 'parabola':
        x = data['x']
        y = data['y']
        p = data['p']
        maxX = data['maxX']
        points = parabola_curve(x,y,p,maxX)
    elif method == 'hyperbola':
        x = data['x']
        y = data['y']
        a = data['a']
        b = data['b']
        maxX = data['maxX']
        points = hyperbola_curve(x,y,a,b,maxX)
    emit('draw', {'points': points})    

@socketio.on('generate_interpolation')
def handle_draw_interpolation(data):
    print(data)
    points = []
    method = data['method']
    if method == 'ermit':
        # Gn = [[P1x, P1y],[P4x, P4y],[R1x, R1y],[R4x, R4y]]
        G = data['G']
        vectors = bresenham_line(G[0][0],G[0][1],G[2][0],G[2][1])
        vectors += bresenham_line(G[1][0],G[1][1],G[3][0],G[3][1])
        points = ermit_interpolation(G)     
    elif method== 'beze':
        # Gb = [[P1x, P1y],[P2x, P2y],[P3x, P3y],[P4x, P4y]]
        G = data['G']   
        vectors = bresenham_line(G[0][0],G[0][1],G[1][0],G[1][1])
        vectors += bresenham_line(G[2][0],G[2][1],G[3][0],G[3][1])
        points = beze_interpolation(G)
    elif method == 'bspline':
        G = data['G']
        vectors = []
        for i in range(0,len(G)-1):
            vectors += bresenham_line(G[i][0],G[i][1],G[i+1][0],G[i+1][1])
        points = bspline_interpolation(G)
    return {'points':points,'vectors':vectors}

@socketio.on('generate_3dfigure')
def handle_3dfigure(data):
    try:
        points = []
        coords = data['coords']
        edges=data['edges']
        rotation = data['r']
        scale = data['s']
        translation = data['t']
        mirror=data['m']
        d=data['d']
        def T(t):
            return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[t[0],t[1],t[2],1]]
        def Rx(rx):
            return [[1,0,0,0],[0,math.cos(math.radians(rx)),math.sin(math.radians(rx)),0],[0,-math.sin(math.radians(rx)),math.cos(math.radians(rx)),0],[0,0,0,1]]
        def Ry(ry):
            return [[math.cos(math.radians(ry)),0,-math.sin(math.radians(ry)),0],[0,1,0,0],[math.sin(math.radians(ry)),0,math.cos(math.radians(ry)),0],[0,0,0,1]]
        def Rz(rz):
            return [[math.cos(math.radians(rz)),math.sin(math.radians(rz)),0,0],[-math.sin(math.radians(rz)),math.cos(math.radians(rz)),0,0],[0,0,1,0],[0,0,0,1]]
        def S(s):
            return [[s[0],0,0,0],[0,s[1],0,0],[0,0,s[2],0],[0,0,0,1]]
        def M(m):
            return [[m[0],0,0,0],[0,m[1],0,0],[0,0,m[2],0],[0,0,0,1]]

        K = multiply_matrices(Rx(rotation[0]),Ry(rotation[1]))
        K = multiply_matrices(K,Rz(rotation[2]))
        K = multiply_matrices(K,S(scale))
        K = multiply_matrices(K,T(translation))
        K = multiply_matrices(K,M(mirror))
        coords = multiply_matrices(coords,K)
        matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,1/d],[0,0,0,1]]
        output = multiply_matrices(coords,matrix)
        for c in output:
            c[0]/=c[3]
            c[0]+=400
            c[1]/=c[3]
            c[1]+=300
            c[2]/=c[3]
        for edge in edges:
            points += bresenham_line(output[edge[0]][0],output[edge[0]][1],output[edge[1]][0],output[edge[1]][1])

        return {'points':points}
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return {'error': "Произошла неизвестная ошибка"}

@socketio.on('generate_polygon')
def handle_polygon(data):
    G = data['G']
    points = []
    method = data['method']

    if method == 'general':
        for i in range(len(G)):
            a = G[i]
            b = G[(i + 1) % len(G)]
            points += bresenham_line(a[0], a[1], b[0], b[1])

    elif method == 'graham':
        start = min(G, key=lambda p: (p[1], p[0]))
        def polar_angle(p):
            return math.atan2(p[1] - start[1], p[0] - start[0])

        def distance(p):
            return (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2

        G.sort(key=lambda p: (polar_angle(p), distance(p)))
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        hull = []
        for p in G:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)

        G = hull

        for i in range(len(G)):
            a = G[i]
            b = G[(i + 1) % len(G)]
            points += bresenham_line(a[0], a[1], b[0], b[1])

    elif method == "jarvis":
        start = min(G, key=lambda p: (p[1], p[0]))
        def polar_angle(p):
            return math.atan2(p[1] - start[1], p[0] - start[0])

        def distance(p):
            return (p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2

        G.sort(key=lambda p: (polar_angle(p), distance(p)))
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        hull = []
        for p in G:
            while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)

        G = hull

        for i in range(len(G)):
            a = G[i]
            b = G[(i + 1) % len(G)]
            points += bresenham_line(a[0], a[1], b[0], b[1])
    
    # Проверка выпуклости
    def is_convex(poly):
        n = len(poly)
        sign = 0
        for i in range(n):
            dx1 = poly[(i + 1) % n][0] - poly[i][0]
            dy1 = poly[(i + 1) % n][1] - poly[i][1]
            dx2 = poly[(i + 2) % n][0] - poly[(i + 1) % n][0]
            dy2 = poly[(i + 2) % n][1] - poly[(i + 1) % n][1]
            z_cross = dx1 * dy2 - dy1 * dx2
            if z_cross != 0:
                if sign == 0:
                    sign = 1 if z_cross > 0 else -1
                elif (z_cross > 0 and sign < 0) or (z_cross < 0 and sign > 0):
                    return False
        return True

    convex = is_convex(G)

    def normalize(vx, vy):
        norm = math.hypot(vx, vy)
        return (vx / norm, vy / norm) if norm != 0 else (0, 0)

    def inward_normal(p1, p2, center):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        nx, ny = -dy, dx  # нормаль влево
        nx, ny = normalize(nx, ny)
        # проверка: направлена ли внутрь
        midx, midy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        testx, testy = midx + nx * 5, midy + ny * 5
        inward = (testx - center[0]) ** 2 + (testy - center[1]) ** 2 < (midx - center[0]) ** 2 + (midy - center[1]) ** 2
        if inward:
            return nx, ny
        return -nx, -ny
     # Центр тяжести
    cx = sum(p[0] for p in G) / len(G)
    cy = sum(p[1] for p in G) / len(G)
    center = (cx, cy)
     # Добавляем нормали
    normal_length = 15
    for i in range(len(G)):
        a = G[i]
        b = G[(i + 1) % len(G)]
        nx, ny = inward_normal(a, b, center)
        mx = (a[0] + b[0]) / 2
        my = (a[1] + b[1]) / 2
        ex = int(mx + nx * normal_length)
        ey = int(my + ny * normal_length)
        points += bresenham_line(int(mx), int(my), ex, ey)

    return {'points':points, 'convex': convex , 'ref_points': G}
        
@socketio.on('check_polygon_intersection')
def handle_polygon_intersection(data):
    def ccw(A, B, C):
    # Проверка на ориентацию (поворот против часовой стрелки)
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def line_intersection(A, B, C, D):
        # Найдём точку пересечения отрезков AB и CD (если существует)
        a1, b1 = A
        a2, b2 = B
        c1, d1 = C
        c2, d2 = D

        denom = (a1 - a2) * (d1 - d2) - (b1 - b2) * (c1 - c2)
        if denom == 0:
            return None  # параллельны или совпадают

        px = ((a1 * b2 - b1 * a2) * (c1 - c2) - (a1 - a2) * (c1 * d2 - d1 * c2)) / denom
        py = ((a1 * b2 - b1 * a2) * (d1 - d2) - (b1 - b2) * (c1 * d2 - d1 * c2)) / denom
        return (px, py)
    
    polygons = data['polygons']
    segment = data['segment']
    intersections = []

    for polygon in polygons:
        A, B = segment
        
        n = len(polygon)
        for i in range(n):
            C = polygon[i]
            D = polygon[(i + 1) % n]  # замкнутость
            if segments_intersect(A, B, C, D):
                point = line_intersection(A, B, C, D)
                if point:
                    intersections.append(point)

    return {'intersections': intersections}

@socketio.on('check_polygon_inside')
def f(data):
    point = data['point']
    polygons = data['polygons']
    inside = False
    print(point)
    print(polygons)
    for polygon in polygons:
        x, y = point        
        n = len(polygon)
        inside = False

        for i in range(n):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % n]

            # Проверка: луч проходит между y0 и y1
            if ((y0 > y) != (y1 > y)):
                # Находим x-координату пересечения стороны с лучом
                x_intersect = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-10) + x0
                if x < x_intersect:
                    inside = not inside
        if inside:
            return {'inside':inside}

    return {'inside':inside}

@socketio.on('generate_filling_polygon')
def handle_filling_polygon(data):
    polygon = data['G']
    method = data['method']
    points = []
    if method == 'scanline':
        points = scanline(polygon)
        
    if method == 'scanline_aet':
        points = scanline_aet(polygon)        
    
    if method == 'flood_fill':
        points = flood_fill(polygon, 800, 600)        

    if method == 'flood_fill_line':
        points = flood_fill_line(polygon, 800, 600)

    return {'points': points} 

@socketio.on('triangulation_voronoi')
def f(data):
    coords = data['points']
    method = data['method']
    points = []
    if method == "triangulation":
        points = delaunay_triangulation(coords)
    elif method == 'voronoi':
        points = voronoi_segments_in_rectangle(coords, 800, 600)

    return {'points': points} 

@socketio.on("remove_invisible_lines")
def f(data):
    lines = data["lines"]
    rectangle = data["rect"]
    a, b = rectangle
    points, new_lines = cohen_sutherland_clip(a,b,lines)

    return {'points':points, "lines":new_lines}

@socketio.on("remove_invisible_faces")
def f(data):
    try:
        points = []
        rotation = data['r']
        is_hidden = data["is_hidden"]
        # Углы в градусах
        rx_deg, ry_deg, rz_deg = rotation
        # Конвертируем в радианы
        rotation = (math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg))
        points = get_drawn_edges_coords(rotation, is_hidden)
    except Exception as e:
        print('Ошибка:',e)

    return {'points':points}

@socketio.on_error_default
def default_error_handler(e):
    print('An error occurred:', e)

if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)
