import tkinter as tk
import math

# Настройки окна и холста
root = tk.Tk()

root.title("3D Cube Viewer")

canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

# Кнопка переключения отображения скрытых граней
show_hidden_faces = tk.BooleanVar(value=True)


def toggle_faces():
    show_hidden_faces.set(not show_hidden_faces.get())
    btn.config(text="Показать все грани" if not show_hidden_faces.get() else "Скрыть невидимые грани")
    draw_cube()

btn = tk.Button(root, text="Скрыть невидимые грани", command=toggle_faces)
btn.pack()

# Куб
cube = {
    "size": 100,
    "position": [0, 0, 0],
    "rotation": [0.5, -0.5, 0.5]
}

is_dragging = False
last_mouse_pos = [0, 0]

def rotate_point(p, rotation):
    x, y, z = p
    rx, ry, rz = rotation

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

drawn_edges_coords = []

def draw_cube():
    global drawn_edges_coords
    drawn_edges_coords = []  # очищаем перед отрисовкой
    canvas.delete("all")
    size = cube["size"] / 2

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

    rotated = [rotate_point(v, cube["rotation"]) for v in vertices]
    projected = [[400 + x, 300 - y, z] for x, y, z in rotated]

    faces = [
        {"vertices": [0, 1, 2, 3], "normal": [0, 0, -1], "color": "#6495ED"},
        {"vertices": [4, 5, 6, 7], "normal": [0, 0, 1],  "color": "#4169E1"},
        {"vertices": [1, 5, 6, 2], "normal": [1, 0, 0],  "color": "#4169E1"},
        {"vertices": [0, 4, 7, 3], "normal": [-1, 0, 0], "color": "#4682B4"},
        {"vertices": [3, 2, 6, 7], "normal": [0, 1, 0],  "color": "#1E90FF"},
        {"vertices": [0, 1, 5, 4], "normal": [0, -1, 0], "color": "#6495ED"}
    ]

    # Вычисляем среднюю глубину каждой грани
    for face in faces:
        z_sum = sum(projected[i][2] for i in face["vertices"])
        face["avg_z"] = z_sum / len(face["vertices"])

    # Сортируем грани по глубине (от дальних к ближним)
    faces_sorted = sorted(faces, key=lambda f: f["avg_z"], reverse=True)

    # 1. Рисуем залитые грани (без контура)
    for face in faces_sorted:
        n = rotate_point(face["normal"], cube["rotation"])
        dot = n[2] * -1  # Вектор взгляда (0, 0, -1)
        visible = dot > 0

        if show_hidden_faces.get() or visible:
            pts = [projected[i][:2] for i in face["vertices"]]
            canvas.create_polygon(pts, fill=face["color"], outline="")  # Без контура

    # 2. Рисуем ребра (контуры) граней в обратном порядке (от ближних к дальним)
    drawn_edges = set()

    for face in reversed(faces_sorted):
        n = rotate_point(face["normal"], cube["rotation"])
        dot = n[2] * -1
        visible = dot > 0

        if show_hidden_faces.get() or visible:
            verts = face["vertices"]
            for i in range(len(verts)):
                v1 = verts[i]
                v2 = verts[(i + 1) % len(verts)]
                edge_key = tuple(sorted((v1, v2)))
                if edge_key not in drawn_edges:
                    drawn_edges.add(edge_key)
                    p1 = projected[v1][:2]
                    p2 = projected[v2][:2]
                    canvas.create_line(*p1, *p2, fill="black", width=2)
                    # Сохраняем координаты ребра
                    drawn_edges_coords.append((p1, p2))
    print(f"{drawn_edges_coords}\n")

# === Обработка мыши ===

def on_mouse_down(event):
    global is_dragging, last_mouse_pos
    is_dragging = True
    last_mouse_pos = [event.x, event.y]

def on_mouse_up(event):
    global is_dragging
    is_dragging = False

def on_mouse_move(event):
    global last_mouse_pos
    if not is_dragging:
        return
    dx = event.x - last_mouse_pos[0]
    dy = event.y - last_mouse_pos[1]
    cube["rotation"][1] += dx * 0.01
    cube["rotation"][0] += dy * 0.01
    last_mouse_pos = [event.x, event.y]
    draw_cube()

canvas.bind("<ButtonPress-1>", on_mouse_down)
canvas.bind("<ButtonRelease-1>", on_mouse_up)
canvas.bind("<B1-Motion>", on_mouse_move)

# Первая отрисовка
draw_cube()
root.mainloop()
