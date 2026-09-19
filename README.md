# Interactive Point Classifier

An educational Python/Tkinter application for exploring a learned decision surface. Left click to add a **red** training example; right click to add a **blue** training example. The colored grid updates as the model learns from the points. Markers show the labels you entered.

Originally developed as part of a Python neural-network course and later cleaned up for publication.

**Author:** Radomyr Karpan

## Model

The implementation has **three independent sigmoid units and no hidden layer**. It is a weighted classifier over polynomial coordinate features, not a deep neural network. Each unit has ten weights for these normalized-position features:

```text
x, y, 1, x², y², x·y, x+y, x²+y², x²·y, y²·x
```

`1` is the bias feature. The combined features allow curved boundaries in the original two-dimensional canvas, even though each unit is linear in its ten inputs.

Every unit is trained against the **same binary target**: 0 for red examples and 1 for blue examples. The three unit outputs become the grid's RGB channels for visualization. They are not three predicted classes, and the background is not intended to reproduce the red and blue marker colors. Lower outputs tend toward dark colors; higher outputs tend toward light colors. The application does not calculate or display an explicit class label.

## Training

Weights start with random values. On each GUI update, the program processes each stored example once, in insertion order. It computes a sigmoid output, the prediction-minus-target error, and the sigmoid derivative `p × (1 − p)`, then updates each weight with per-example gradient descent on squared error. The learning rate is 2. The grid is drawn before that update pass, so it shows the model state from immediately before the latest training pass.

## Requirements

Python 3 with Tkinter support. The program uses only the Python standard library; no package installation or `requirements.txt` is needed.

## Run

From this directory:

```powershell
python neuClasification.py
```

Click the canvas to add examples. Close the window to stop the program.

## Project structure

- `neuClasification.py` — Tkinter interface, model, and training loop
- `.gitignore` — Python and editor-generated files
