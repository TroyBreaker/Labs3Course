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
'''label_x1 = tk.Label(root, text="x1:")
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
entry_y2.pack()'''

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

        # Определяем грани (треугольники)
        faces = [
            [0,1,2], [1,3,2],        # основание
            [0,1,4], [1,3,4], [3,2,4], [2,0,4]  # боковые грани
        ]

        for edge in self.edges:
            line(output[edge[0]][0],output[edge[0]][1],output[edge[1]][0],output[edge[1]][1])

        
coords = [[-100,100,0,1],[100,100,0,1],[-100,-100,0,1],[100,-100,0,1],
[-100,100,100,1],[100,100,100,1],[-100,-100,100,1],[100,-100,100,1]]
edges =[[0,1],[0,2],[0,4],
[3,1],[3,2],[3,7],
[6,4],[6,2],[6,7],
[5,4],[5,1],[5,7]]

T = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
rx = 120
Rx=[[1,0,0,0],[0,math.cos(math.radians(rx)),math.sin(math.radians(rx)),0],[0,-math.sin(math.radians(rx)),math.cos(math.radians(rx)),0],[0,0,0,1]]
ry = 0
Ry=[[math.cos(math.radians(ry)),0,-math.sin(math.radians(ry)),0],[0,1,0,0],[math.sin(math.radians(ry)),0,math.cos(math.radians(ry)),0],[0,0,0,1]]
'''Rz=[[math.cos(math.radians(45)),math.sin(math.radians(45)),0,0],[-math.sin(math.radians(45)),math.cos(math.radians(45)),0,0],[0,0,1,0],[0,0,0,1]]
S=[[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0],[0,0,0,1]]
M=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'''

coords = multiply_matrices(coords,Rx)
coords = multiply_matrices(coords,Ry)
coords = multiply_matrices(coords,T)

fig = Figure(coords,edges)
# fig.draw()

'''a = [([400.93393998192585, 346.65965452983266], [480.70318910895475, 303.08151511876713]), ([480.70318910895475, 303.08151511876713], [427.54451778501675, 219.6282801902583]), ([427.54451778501675, 219.6282801902583], [347.7752686579879, 263.2064196013238]), ([347.7752686579879, 263.2064196013238], [400.93393998192585, 346.65965452983266]), ([400.93393998192585, 346.65965452983266], [372.45548221498325, 380.3717198097417]), ([372.45548221498325, 380.3717198097417], [319.29681089104525, 296.91848488123287]), ([319.29681089104525, 296.91848488123287], [347.7752686579879, 263.2064196013238]), ([480.70318910895475, 303.08151511876713], [452.2247313420121, 336.7935803986762]), ([452.2247313420121, 336.7935803986762], [372.45548221498325, 380.3717198097417])]

for i in a:
    x0,y0 = i[0]
    x1,y1 = i[1]
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    line(x0,y0,x1,y1)
'''
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

import numpy as np
from scipy.spatial import Delaunay

def delaunay_triangulation(points):
    points_np = np.array(points)
    tri = Delaunay(points_np)
    # Преобразуем каждый треугольник из numpy.array в обычный список
    triangles_coords = [points_np[triangle].tolist() for triangle in tri.simplices]
    return triangles_coords

example_points = [(0, 0), (100, 0), (320, 200), (0, 100),(250,300),(100,600)]
triangles = delaunay_triangulation(example_points)

'''for tri in triangles:
    for i in range(3):
        x0, y0 = tri[i]
        x1, y1 = tri[(i+1)%3]
        points = line(x0, y0, x1, y1)'''



'''import numpy as np
from scipy.spatial import Voronoi
def voronoi_segments_in_rectangle(points, a, b):
    vor = Voronoi(points)
    segments = []

    # Прямоугольная граница
    xmin, xmax = 0, a
    ymin, ymax = 0, b

    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue  # Пропустить бесконечные ребра

        p1 = vor.vertices[ridge_vertices[0]]
        p2 = vor.vertices[ridge_vertices[1]]

        # Оставить только те, что находятся внутри прямоугольника
        if all([
            xmin <= p1[0] <= xmax,
            ymin <= p1[1] <= ymax,
            xmin <= p2[0] <= xmax,
            ymin <= p2[1] <= ymax
        ]):
            segments.append((tuple(p1), tuple(p2)))

    return segments
'''



from scipy.spatial import Voronoi

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
    vor = Voronoi(points)
    segments = []

    xmin, xmax = 0, a
    ymin, ymax = 0, b

    # Контур прямоугольника
    rect = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]

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

    return segments, rect

# Пример использования:
num_points = 8
a, b = 800, 600
points = np.random.rand(num_points, 2) * [a, b]



segments, w = voronoi_segments_in_rectangle(points, a, b)

'''for point in points:
    x, y =point
    canvas.create_oval(x, y, x + 3, y+ 3, fill="black")
for seg in segments:
    x0, y0 = seg[0]
    x1, y1 = seg[1]
    line(x0, y0, x1, y1)'''



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

    return clipped_segments

rect_point1 = [100, 50]
rect_point2 = [400, 300]

segments = [
    [[0, 0], [600, 600]],     # частично входит
    [[200, 200], [100, 400]],     # полностью внутри
    [[0, 300], [300, 0]],     # частично входит
    [[600, 600], [700, 700]]      # полностью вне
]

segments = cohen_sutherland_clip(rect_point1, rect_point2, segments)

'''line(rect_point1[0],rect_point1[1],rect_point2[0],rect_point1[1])
line(rect_point2[0],rect_point1[1],rect_point2[0],rect_point2[1])print(segments)
for seg in segments:
    x0, y0 = seg[0]
    x1, y1 = seg[1]
    line(x0, y0, x1, y1)'''


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

# Углы в градусах
rx_deg, ry_deg, rz_deg = 45, 45, 45
# Конвертируем в радианы
rotation = (math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg))

edges_to_draw = get_drawn_edges_coords(rotation, show_hidden_faces=True)

for seg in edges_to_draw:
    x0, y0 = seg[0]
    x1, y1 = seg[1]
    canvas.create_line(x0, y0, x1, y1)



# Создаем кнопку для рисования прямой
'''button = tk.Button(root, text="Нарисовать эллипс", command=BresenhamCircle)
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
button.pack()'''

# Запускаем главный цикл обработки событий
root.mainloop()
