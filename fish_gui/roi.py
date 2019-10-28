import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')

class Model():
    def __init__(self, images):
        self.images = images
        self.history = [next(images)]
        self.random = []
        self.cursor = 0

    @property
    def rand_max(self):
        """randomly overlapping 10 images"""
        step = 1
        if not self.random:
            while True:
                for _ in range(step):
                    try:
                        img = next(self.images)
                    except StopIteration:
                        img = None
                        break
                if not isinstance(img, type(None)):
                    self.random.append(img)
                    step *= 2
                else:
                    break
        return np.max(self.random, axis=0)

    @property
    def rand_min(self):
        """randomly overlapping 10 images"""
        step = 1
        if not self.random:
            while True:
                for _ in range(step):
                    try:
                        img = next(self.images)
                    except StopIteration:
                        img = None
                        break
                if not isinstance(img, type(None)):
                    self.random.append(img)
                    step *= 2
                else:
                    break
        return np.min(self.random, axis=0)


    @property
    def max(self):
        return np.max(list(self.images), axis=0)

    @property
    def min(self):
        return np.min(list(self.images), axis=0)

    @property
    def next(self):
        if self.cursor == len(self.history) - 1:
            img = next(self.images)
            self.history.append(img)
            self.cursor += 1
            return img
        else:
            img = self.history[self.cursor]
            self.cursor += 1
            return img

    @property
    def back(self):
        self.cursor -= 1
        self.cursor = max(self.cursor, 0)
        return self.history[self.cursor]

class Viewer(QMainWindow):
    def __init__(self, images, roi):
        super().__init__()
        self.model = Model(images)
        self.layout = QGridLayout()
        self.window = QWidget()
        self.roi = roi
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_canvas()
        self.__setup_control()
        self.show()

    def __setup_canvas(self):
        image = self.model.next

        window = pg.GraphicsLayoutWidget(border=True)
        window.setWindowTitle('Measuring ROI')


        view = window.addViewBox(row=0, col=0, lockAspect=True)

        view.setRange(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        view.setLimits(xMin=0, yMin=0, xMax=image.shape[1], yMax=image.shape[0])
        canvas = pg.ImageItem()
        canvas.setImage(image)
        self.roi_widget = pg.RectROI([self.roi[0], self.roi[1]], [self.roi[2], self.roi[3]], pen=(0 ,9))

        view.addItem(canvas)
        view.addItem(self.roi_widget)

        self.canvas = canvas
        self.layout.addWidget(window, 0, 0)

    def __setup_control(self):
        pannel = QWidget()
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.btn_max = QPushButton('Random Max')
        self.btn_min = QPushButton('Random Min')
        self.btn_save = QPushButton('Save')
        self.btn_exit = QPushButton('Exit')
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_exit)

        self.btn_max.clicked.connect(self.get_max)
        self.btn_min.clicked.connect(self.get_min)
        self.btn_save.clicked.connect(self.save)
        self.btn_exit.clicked.connect(self.close)

        self.layout.addWidget(pannel, 1, 0)

    def get_max(self):
        image = self.model.rand_max
        self.canvas.setImage(image)

    def get_min(self):
        image = self.model.rand_min
        self.canvas.setImage(image)

    def save(self):
        position = np.array(self.roi_widget.pos()).astype(int)
        size = np.array(self.roi_widget.size()).astype(int)
        self.roi = position.tolist() + size.tolist()
        self.close()

def measure_roi(images, roi):
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    measure = Viewer(images, roi)
    app.exec_()
    return measure.roi

if __name__ == "__main__":
    import fish_track as ft
    path = '/media/yushi/hdd/observe_20190502/5_fish_swarm/cam-1'
    images = ft.read.iter_image_sequence(path)
    measure_roi(images, [0, 100, 0, 100])
