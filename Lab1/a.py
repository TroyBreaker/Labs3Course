import math

def get_drawn_edges_coords(rotation_angles, show_hidden_faces):
    """
    rotation_angles: (rx, ry, rz) в радианах
    show_hidden_faces: bool - показывать все грани или скрывать невидимые

    Возвращает список ребер [(p1, p2), ...], где p1 и p2 - 2D координаты (x, y)
    """
    rx, ry, rz = rotation_angles
    size = 100 / 2

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

    return edges_coords

import math

# Углы в градусах
rx_deg, ry_deg, rz_deg = 0, 40, 0
# Конвертируем в радианы
rotation = (math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg))

edges_to_draw = get_drawn_edges_coords(rotation, show_hidden_faces=False)

for p1, p2 in edges_to_draw:
    print(f"Edge from {p1} to {p2}")
