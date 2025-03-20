import matplotlib.pyplot as plt

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
plt.show()