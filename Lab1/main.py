'''# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys


# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Paint with PyQt5")

        # setting geometry to main window
        self.setGeometry(100, 100, 800, 600)

        # creating image object
        self.image = QImage(self.size(), QImage.Format_RGB32)

        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 2
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()

        # creating menu bar
        mainMenu = self.menuBar()

        # creating file menu for save and clear action
        fileMenu = mainMenu.addMenu("File")

        # adding brush size to main menu
        b_size = mainMenu.addMenu("Brush Size")

        # adding brush color to ain menu
        b_color = mainMenu.addMenu("Brush Color")

        # creating save action
        saveAction = QAction("Save", self)
        # adding short cut for save action
        saveAction.setShortcut("Ctrl + S")
        # adding save to the file menu
        fileMenu.addAction(saveAction)
        # adding action to the save
        saveAction.triggered.connect(self.save)

        # creating clear action
        clearAction = QAction("Clear", self)
        # adding short cut to the clear action
        clearAction.setShortcut("Ctrl + C")
        # adding clear to the file menu
        fileMenu.addAction(clearAction)
        # adding action to the clear
        clearAction.triggered.connect(self.clear)

        # creating options for brush sizes
        # creating action for selecting pixel of 4px
        pix_4 = QAction("4px", self)
        # adding this action to the brush size
        b_size.addAction(pix_4)
        # adding method to this
        pix_4.triggered.connect(self.Pixel_4)

        # similarly repeating above steps for different sizes
        pix_7 = QAction("7px", self)
        b_size.addAction(pix_7)
        pix_7.triggered.connect(self.Pixel_7)

        pix_9 = QAction("9px", self)
        b_size.addAction(pix_9)
        pix_9.triggered.connect(self.Pixel_9)

        pix_12 = QAction("12px", self)
        b_size.addAction(pix_12)
        pix_12.triggered.connect(self.Pixel_12)

        # creating options for brush color
        # creating action for black color
        black = QAction("Black", self)
        # adding this action to the brush colors
        b_color.addAction(black)
        # adding methods to the black
        black.triggered.connect(self.blackColor)

        # similarly repeating above steps for different color
        white = QAction("White", self)
        b_color.addAction(white)
        white.triggered.connect(self.whiteColor)

        green = QAction("Green", self)
        b_color.addAction(green)
        green.triggered.connect(self.greenColor)

        yellow = QAction("Yellow", self)
        b_color.addAction(yellow)
        yellow.triggered.connect(self.yellowColor)

        red = QAction("Red", self)
        b_color.addAction(red)
        red.triggered.connect(self.redColor)

    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())

            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if filePath == "":
            return
        self.image.save(filePath)

    # method for clearing every thing on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    # methods for changing pixel sizes
    def Pixel_4(self):
        self.brushSize = 4

    def Pixel_7(self):
        self.brushSize = 7

    def Pixel_9(self):
        self.brushSize = 9

    def Pixel_12(self):
        self.brushSize = 12

    # methods for changing brush color
    def blackColor(self):
        self.brushColor = Qt.black

    def whiteColor(self):
        self.brushColor = Qt.white

    def greenColor(self):
        self.brushColor = Qt.green

    def yellowColor(self):
        self.brushColor = Qt.yellow

    def redColor(self):
        self.brushColor = Qt.red


# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the window
window.show()

# start the app
sys.exit(App.exec())
'''

import tkinter as tk
import math

def draw_line():
    x1, y1 = int(entry_x1.get()), int(entry_y1.get())
    x2, y2 = int(entry_x2.get()), int(entry_y2.get())

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    err = dx - dy

    while True:
        canvas.create_oval(x1, y1, x1 + 1, y1 + 1, fill="black")
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def BresenhamCircle():
    _x, _y = int(entry_x1.get()), int(entry_y1.get())
    a, b = abs(int(entry_x2.get())), abs(int(entry_y2.get()))
    x = 0
    y = b
    delta = a*a+b*b-2*a*a*b
    
    while y >= 0:
        x1 = _x + x
        y1 = _y + y
        canvas.create_oval(x1, y1, x1 + 1, y1 + 1, fill="black")
        
        x1 = _x + x
        y1 = _y - y
        canvas.create_oval(x1, y1, x1 + 1, y1 + 1, fill="black")
        
        x1 = _x - x
        y1 = _y - y
        canvas.create_oval(x1, y1, x1 + 1, y1 + 1, fill="black")
        
        x1 = _x - x
        y1 = _y + y
        canvas.create_oval(x1, y1, x1 + 1, y1 + 1, fill="black")
        
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

def drawParabola():
    x0, y0 = int(entry_x1.get()), int(entry_y1.get())
    p, maxX = int(entry_x2.get()), int(entry_y2.get())
    x = 0
    y = 0
    Sd = (y + 1) ** 2 - 2 * abs(p) * (x + 1)
    Sv = (y + 1) ** 2 - 2 * abs(p) * x
    Sh = y ** 2 - 2 * abs(p) * (x + 1)
    
    canvas.create_oval(x0 + x, y0 + y, x0 + x + 1, y0 + y + 1, fill="black")
    
    while x < maxX:
        if abs(Sh) - abs(Sv) <= 0:
            if abs(Sd) - abs(Sh) < 0:
                y += 1
            x += 1
        else:
            if abs(Sv) - abs(Sd) > 0:
                x += 1
            y += 1
        if p>0:
            canvas.create_oval(x0 + x, y0 + y, x0 + x + 1, y0 + y + 1, fill="black")
            canvas.create_oval(x0 + x, y0 - y, x0 + x + 1, y0 - y + 1, fill="black")
        if p<0:
            canvas.create_oval(x0 - x, y0 + y, x0 - x + 1, y0 + y + 1, fill="black")
            canvas.create_oval(x0 - x, y0 - y, x0 - x + 1, y0 - y + 1, fill="black")
        Sd = (y + 1) ** 2 - 2 * abs(p) * (x + 1)
        Sv = (y + 1) ** 2 - 2 * abs(p) * x
        Sh = y ** 2 - 2 * abs(p) * (x + 1)

def drawParabola2():
    x0, y0 = int(entry_x1.get()), int(entry_y1.get())
    p, maxY = int(entry_x2.get()), int(entry_y2.get())
    x = 0
    y = 0

    Sd = (1/abs(p)) * (x + 1)*(x + 1) - (y + 1)
    Sv = (1/abs(p))* x*x - (y +1)
    Sh = (1/abs(p)) * (x + 1)*(x + 1) - y
    
    canvas.create_oval(x0, y0, x0 + 1, y0+ 1, fill="black")
    
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
            canvas.create_oval(x0 + x, y0 - y, x0 + x + 1, y0 - y + 1, fill="black")
            canvas.create_oval(x0 - x, y0 - y, x0 - x + 1, y0 - y + 1, fill="black")
        if p<0:
            canvas.create_oval(x0 + x, y0 + y, x0 + x + 1, y0 + y + 1, fill="black")
            canvas.create_oval(x0 - x, y0 + y, x0 - x + 1, y0 + y + 1, fill="black")
        Sd = (1/abs(p)) * (x + 1)*(x + 1) - (y + 1)
        Sv = (1/abs(p)) * x*x - (y + 1)
        Sh = (1/abs(p)) * (x + 1)*(x + 1) - y

def drawHyperbola():
    x0, y0 = int(entry_x1.get()), int(entry_y1.get())
    a, b = int(entry_x2.get()), int(entry_y2.get())
    x0 -= abs(a)
    x = abs(a)
    y = 0
    Sd = 1-(x+1)**2/a**2+(y+1)**2/b**2
    Sv = 1-x**2/a**2+(y+1)**2/b**2
    Sh = 1-(x+1)**2/a**2+y**2/b**2
    
    canvas.create_oval(x0 + x, y0 + y, x0 + x + 1, y0 + y + 1, fill="black")
    
    while x < 100+a:
        if abs(Sh) - abs(Sv) <= 0:
            if abs(Sd) - abs(Sh) < 0:
                y += 1
            x += 1
        else:
            if abs(Sv) - abs(Sd) > 0:
                x += 1
            y += 1
        canvas.create_oval(x0 + x, y0 + y, x0 + x + 1, y0 + y + 1, fill="black")
        canvas.create_oval(x0 + x, y0 - y, x0 + x + 1, y0 - y + 1, fill="black")
        '''canvas.create_oval(x0 - x, y0 + y, x0 - x + 1, y0 + y + 1, fill="black")
        canvas.create_oval(x0 - x, y0 - y, x0 - x + 1, y0 - y + 1, fill="black")'''
        Sd = 1-(x+1)**2/a**2+(y+1)**2/b**2
        Sv = 1-x**2/a**2+(y+1)**2/b**2
        Sh = 1-(x+1)**2/a**2+y**2/b**2
    a = 0

def ErmitCurve():
    # P1x, P1y = int(entry_x1.get()), int(entry_y1.get())
    # R1x, R1y = int(entry_x2.get()), int(entry_y2.get())
    a=1800
    P1x, P1y = 0,0
    R1x, R1y = a, a
    P4x, P4y = 500, 0
    R4x, R4y = 2300, -a
    Mn = [[2,-2,1,1],[-3,3,-2,-1],[0,0,1,0],[1,0,0,0]]
    Gn = [[P1x, P1y],[P4x, P4y],[R1x, R1y],[R4x, R4y]]
    for i in range(0,len(Gn[0]),1):
        Gn[2][i]-=Gn[0][i]
        Gn[3][i]-=Gn[1][i]

    C = multiply_matrices(Mn,Gn)

    points = []

    for t in range(0,21,1):
        t/=20.0
        T = [[t**3,t**2,t,1]]
        points.append(multiply_matrices(T,C))

    for i in range(0,len(points)-1):
        line(points[i][0][0], points[i][0][1], points[i+1][0][0], points[i+1][0][1])

def Beze():
    P1x, P1y = 0,500
    P2x, P2y = 500, 0
    P3x, P3y = 600, 600
    P4x, P4y = 300, 400
    Mb = [[-1,3,-3,1],[3,-6,3,0],[-3,3,0,0],[1,0,0,0]]
    Gb = [[P1x, P1y],[P2x, P2y],[P3x, P3y],[P4x, P4y]]

    C = multiply_matrices(Mb,Gb)

    points = []

    for t in range(0,21,1):
        t/=20.0
        T = [[t**3,t**2,t,1]]
        points.append(multiply_matrices(T,C))

    for i in range(0,len(points)-1):
        line(points[i][0][0], points[i][0][1], points[i+1][0][0], points[i+1][0][1])

def Bspline():
    Ms = [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 0, 3, 0], [1, 4, 1, 0]]

    for i in range(len(Ms)):
        for j in range(len(Ms[i])):
            Ms[i][j] *= 1/6


    G = [[200, 0],[400, 0],[400, 200],[400, 400]
    ,[200,400],[0, 400],[0,200],[0,0]
    ,[200,0],[400,0],[400,200]
    ]
    # G = [[0, 0],[100, 300],[200, 100],[300, 300]]
    for i in range(len(G)):
        for j in range(len(G[i])):
            G[i][j] +=50


    for i in range(0,len(G)-3):
        Cs=[G[i],G[i+1],G[i+2],G[i+3]]
        C = multiply_matrices(Ms,Cs)

        points = []

        for t in range(0,21,1):
            t/=20.0
            T = [[t**3,t**2,t,1]]
            points.append(multiply_matrices(T,C))

        for i in range(0,len(points)-1):
            line(points[i][0][0], points[i][0][1], points[i+1][0][0], points[i+1][0][1])

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
    # Создаем пустую матрицу-результат
    result = [[0 for _ in range(rows_m2)] for _ in range(rows_m1)]
    # Умножаем матрицы
    for i in range(rows_m1):
        for j in range(rows_m2):
            for k in range(cols_m1):
                result[i][j] += m1[i][k] * m2[j][k]
    return result

def line(x0,y0,x1,y1):
    length = max(abs(x1-x0),abs(y1-y0))
    dx = (x1-x0)/length
    dy = (y1-y0)/length
    x = x0 + 0.5 * (-1 if dx < 0 else 1)
    y = y0 + 0.5 * (-1 if dy < 0 else 1)
    i = 0

    while i<=length:
        x = x + dx
        y = y + dy
        canvas.create_oval(x, y, x + 1, y+ 1, fill="black")
        i = i + 1

# Создаем окно
root = tk.Tk()
root.title("Рисование прямой между двумя точками")

# Создаем элементы управления для ввода координат точек
label_x1 = tk.Label(root, text="x1:")
label_x1.pack()
entry_x1 = tk.Entry(root)
entry_x1.pack()

label_y1 = tk.Label(root, text="y1:")
label_y1.pack()
entry_y1 = tk.Entry(root)
entry_y1.pack()

label_x2 = tk.Label(root, text="x2:")
label_x2.pack()
entry_x2 = tk.Entry(root)
entry_x2.pack()

label_y2 = tk.Label(root, text="y2:")
label_y2.pack()
entry_y2 = tk.Entry(root)
entry_y2.pack()

# Создаем холст
canvas = tk.Canvas(root, width=800, height=600, bg="white")
canvas.pack()

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

class Figure:
    def __init__(self,coords,edges):
        self.coords = coords
        self.edges=edges
    
    def draw(self):
        d=300
        self.matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,1/d],[0,0,0,1]]
        output = multiply_matrices(self.coords,self.matrix)
        for c in output:
            c[0]/=c[3]
            c[0]+=400
            c[1]/=c[3]
            c[1]+=300
            c[2]/=c[3]
            canvas.create_oval(c[0], c[1], c[0] + 1, c[1]+ 1, fill="black")
        for edge in self.edges:
            line(output[edge[0]][0],output[edge[0]][1],output[edge[1]][0],output[edge[1]][1])
        
coords = [[-100,100,0,1],[100,100,0,1],[-100,-100,0,1],[100,-100,0,1],[0,0,100,1]]
T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[100,150,0,1]]
Rz=[[math.cos(math.radians(45)),math.sin(math.radians(45)),0,0],[-math.sin(math.radians(45)),math.cos(math.radians(45)),0,0],[0,0,1,0],[0,0,0,1]]
Rx=[[1,0,0,0],[0,math.cos(math.radians(90)),math.sin(math.radians(90)),0],[0,-math.sin(math.radians(90)),math.cos(math.radians(90)),0],[0,0,0,1]]
Ry=[[math.cos(math.radians(45)),0,-math.sin(math.radians(45)),0],[0,1,0,0],[math.sin(math.radians(45)),0,math.cos(math.radians(45)),0],[0,0,0,1]]
S=[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,1]]
M=[[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
coords = multiply_matrices(coords,Rx)
coords = multiply_matrices(coords,T)
coords = multiply_matrices(coords,M)

edges =[[0,1],[0,2],[3,1],[3,2], [0,4],[1,4],[2,4],[3,4]]
fig = Figure(coords,edges)
# fig.draw()

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
            canvas.create_oval(x, y, x + 1, y+ 1, fill="black")
            # Добавляем соседние точки в стек (4-связность)
            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

    print(f"Всего добавлено точек: {len(points)}")
    return points

def is_border(x, y, polygon):
    """
    Проверяет, является ли точка (x, y) частью границы полигона.
    """
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (x1 == x2):  # Вертикальный отрезок
            if x == x1 and min(y1, y2) <= y <= max(y1, y2):
                return True
        elif (y1 == y2):  # Горизонтальный отрезок
            if y == y1 and min(x1, x2) <= x <= max(x1, x2):
                return True
        else:
            # Для простоты считаем только оси-ориентированные полигоны
            continue
    return False

def scanline_fill(polygon,width, height):
    """
    Реализует построчный алгоритм заполнения с затравкой.
    polygon: список вершин полигона [(x1, y1), (x2, y2), ...]
    seed: (x, y) — координата затравки
    Возвращает список точек, которые были залиты.
    """
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
        if ()


    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    
    filled = set()  # множество залитых точек
    stack = [seed]

    while stack:
        x, y = stack.pop()

        # Пропускаем если уже залито или на границе
        if (x, y) in filled or is_border(x, y, polygon):
            continue

        # Расширяем интервал влево
        xl = x
        while xl >= min_x and not is_border(xl, y, polygon) and (xl, y) not in filled:
            filled.add((xl, y))
            xl -= 1
        xl += 1  # возвращаемся на последний валидный

        # Расширяем интервал вправо
        xr = x + 1
        while xr <= max_x and not is_border(xr, y, polygon) and (xr, y) not in filled:
            filled.add((xr, y))
            xr += 1
        xr -= 1

        # Проверяем строки сверху и снизу
        for new_y in [y - 1, y + 1]:
            if new_y < min_y or new_y > max_y:
                continue
            in_span = False
            for xi in range(xl, xr + 1):
                if (xi, new_y) not in filled and not is_border(xi, new_y, polygon):
                    if not in_span:
                        stack.append((xi, new_y))
                        in_span = True
                else:
                    in_span = False

    return list(filled)

points = seed_fill_auto([[0,0], [100,0],[320,200], [0,100]],800,600)


# Создаем кнопку для рисования прямой
button = tk.Button(root, text="Нарисовать эллипс", command=BresenhamCircle)
button.pack()
button = tk.Button(root, text="Нарисовать параболу", command=drawParabola2)
button.pack()
button = tk.Button(root, text="Нарисовать гиперболу", command=drawHyperbola)
button.pack()

button = tk.Button(root, text="Эрмит", command=ErmitCurve)
button.pack()

button = tk.Button(root, text="Безье", command=Beze)
button.pack()

button = tk.Button(root, text="B-сплайн", command=Bspline)
button.pack()

# Запускаем главный цикл обработки событий
root.mainloop()
