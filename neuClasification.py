import tkinter as tk
import math
from random import uniform

window = tk.Tk()
window.geometry("500x500")
window.title("classification")
canvas = tk.Canvas(window, width=500, height=500, bg='#000',highlightthickness=0)
canvas.pack()

# Ten weights per output, corresponding to the polynomial features below.
weights = [
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
]

weights1 = [
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
]

weights2 = [
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
    uniform(-1, 1),
]

def sigmoid_derivative(p):
    return p * (1 - p)

def rgb(r, g, b):
    r = math.floor(min(max(r, 0), 255))
    g = math.floor(min(max(g, 0), 255))
    b = math.floor(min(max(b, 0), 255))
    return f'#{r:02x}{g:02x}{b:02x}'

def on_left_click(event):
    data.append({
        'x': event.x,
        'y': event.y,
        'class': 'red',
    })
    circle(event.x, event.y, 'red')

def on_right_click(event):
    data.append({
        'x': event.x,
        'y': event.y,
        'class': 'blue',
    })
    circle(event.x, event.y, 'blue')

def circle(centerx, centery, color, radius=8):
    shift = max(abs(radius),3)/2
    startx = centerx - shift
    starty = centery - shift
    endx = centerx + shift
    endy = centery + shift
    canvas.create_oval(startx, starty, endx, endy, fill=color, width=0)

canvas.bind("<Button-1>", on_left_click)
canvas.bind("<Button-3>", on_right_click)

data = []

rects = []
size = 20

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

for x in range(25):
    rects.append([])
    for y in range(25):
        posx = x * size
        posy = y * size
        width = posx + size
        height = posy + size
        rects[x].append(canvas.create_rectangle(posx, posy, width, height, fill='white', width=0))

def main():
    for x in range(25):
        for y in range(25):
            nx = x*20/250-1
            ny = y*20/250-1
            n1 = nx
            n2 = ny
            n3 = 1
            n4 = nx*nx
            n5 = ny*ny
            n6 = nx*ny
            n7 = nx+ny
            n8 = nx*nx+ny*ny
            n9 = nx*nx*ny
            n10 = ny*ny*nx
            
            out1 = n1*weights[0]
            out1 += n2*weights[1]
            out1 += n3*weights[2]
            out1 += n4*weights[3]
            out1 += n5*weights[4]
            out1 += n6*weights[5]
            out1 += n7*weights[6]
            out1 += n8*weights[7]
            out1 += n9*weights[8]
            out1 += n10*weights[9]

            out2 = n1*weights1[0]
            out2 += n2*weights1[1]
            out2 += n3*weights1[2]
            out2 += n4*weights1[3]
            out2 += n5*weights1[4]
            out2 += n6*weights1[5]
            out2 += n7*weights1[6]
            out2 += n8*weights1[7]
            out2 += n9*weights1[8]
            out2 += n10*weights1[9]

            out3 = n1*weights2[0]
            out3 += n2*weights2[1]
            out3 += n3*weights2[2]
            out3 += n4*weights2[3]
            out3 += n5*weights2[4]
            out3 += n6*weights2[5]
            out3 += n7*weights2[6]
            out3 += n8*weights2[7]
            out3 += n9*weights2[8]
            out3 += n10*weights2[9]
            
            
            out1 = sigmoid(out1)*255
            out2 = sigmoid(out2)*255
            out3 = sigmoid(out3)*255
            color = rgb(out1,out2,out3)
            canvas.itemconfig(rects[x][y], fill=color)

    for example in data:
        x = example['x']/250-1
        y = example['y']/250-1
        n1 = x
        n2 = y
        n3 = 1
        n4 = x*x
        n5 = y*y
        n6 = x*y
        n7 = x+y
        n8 = x*x+y*y
        n9 = x*x*y
        n10 = y*y*x

        out1 = n1*weights[0]
        out1 += n2*weights[1]
        out1 += n3*weights[2]
        out1 += n4*weights[3]
        out1 += n5*weights[4]
        out1 += n6*weights[5]
        out1 += n7*weights[6]
        out1 += n8*weights[7]
        out1 += n9*weights[8]
        out1 += n10*weights[9]

        out2 = n1*weights1[0]
        out2 += n2*weights1[1]
        out2 += n3*weights1[2]
        out2 += n4*weights1[3]
        out2 += n5*weights1[4]
        out2 += n6*weights1[5]
        out2 += n7*weights1[6]
        out2 += n8*weights1[7]
        out2 += n9*weights1[8]
        out2 += n10*weights1[9]

        out3 = n1*weights2[0]
        out3 += n2*weights2[1]
        out3 += n3*weights2[2]
        out3 += n4*weights2[3]
        out3 += n5*weights2[4]
        out3 += n6*weights2[5]
        out3 += n7*weights2[6]
        out3 += n8*weights2[7]
        out3 += n9*weights2[8]
        out3 += n10*weights2[9]
        
        out1 = sigmoid(out1)
        out2 = sigmoid(out2)
        out3 = sigmoid(out3)
        error = 0

        if example['class'] == 'red':
            error = out1 - 0
        elif example['class'] == 'blue':
            error = out1 - 1   
        delta1 = error * sigmoid_derivative(out1)

        if example['class'] == 'red':
            error = out2 - 0
        elif example['class'] == 'blue':
            error = out2 - 1   
        delta2 = error * sigmoid_derivative(out2)

        if example['class'] == 'red':
            error = out3 - 0
        elif example['class'] == 'blue':
            error = out3 - 1   
        delta3 = error * sigmoid_derivative(out3)

        learning_rate = 2
        weights[0] -= delta1 * n1 * learning_rate
        weights[1] -= delta1 * n2 * learning_rate
        weights[2] -= delta1 * n3 * learning_rate
        weights[3] -= delta1 * n4 * learning_rate
        weights[4] -= delta1 * n5 * learning_rate
        weights[5] -= delta1 * n6 * learning_rate
        weights[6] -= delta1 * n7 * learning_rate
        weights[7] -= delta1 * n8 * learning_rate
        weights[8] -= delta1 * n9 * learning_rate
        weights[9] -= delta1 * n10 * learning_rate

        learning_rate = 2
        weights1[0] -= delta2 * n1 * learning_rate
        weights1[1] -= delta2 * n2 * learning_rate
        weights1[2] -= delta2 * n3 * learning_rate
        weights1[3] -= delta2 * n4 * learning_rate
        weights1[4] -= delta2 * n5 * learning_rate
        weights1[5] -= delta2 * n6 * learning_rate
        weights1[6] -= delta2 * n7 * learning_rate
        weights1[7] -= delta2 * n8 * learning_rate
        weights1[8] -= delta2 * n9 * learning_rate
        weights1[9] -= delta2 * n10 * learning_rate

        learning_rate = 2
        weights2[0] -= delta3 * n1 * learning_rate
        weights2[1] -= delta3 * n2 * learning_rate
        weights2[2] -= delta3 * n3 * learning_rate
        weights2[3] -= delta3 * n4 * learning_rate
        weights2[4] -= delta3 * n5 * learning_rate
        weights2[5] -= delta3 * n6 * learning_rate
        weights2[6] -= delta3 * n7 * learning_rate
        weights2[7] -= delta3 * n8 * learning_rate
        weights2[8] -= delta3 * n9 * learning_rate
        weights2[9] -= delta3 * n10 * learning_rate


        
    window.after(10, main)
main()

window.mainloop()
