import fish_3d as f3
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout, QLineEdit
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')

class Model():
    def __init__(self):
        pass

class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Model()

        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_left()
        self.__setup_right()
        self.show()

    def __setup_left(self):
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        window.setWindowTitle('Measuring ROI')

        view = window.addViewBox(row=0, col=0, lockAspect=True)

        #view.setRange(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        #view.setLimits(xMin=0, yMin=0, xMax=image.shape[1], yMax=image.shape[0])
        canvas = pg.ImageItem()
        view.addItem(canvas)

        self.left_canvas = canvas

        self.layout.addWidget(window, 0, 0)

        pannel = QWidget()
        layout = QGridLayout()
        pannel.setLayout(layout)
        self.btn_load_image = QPushButton('Load Image')
        self.btn_load_camera = QPushButton('Load Camera')

        label_interface  = QLabel('Water Level')
        label_normal  = QLabel('Normal Direction')
        label_step  = QLabel('Step')

        self.left_edit_interface  = QLineEdit('0')
        self.left_edit_normal  = QLineEdit('0, 0, 1')
        self.left_edit_step  = QLineEdit('1')

        layout.addWidget(self.btn_load_image, 0, 0)
        layout.addWidget(self.btn_load_camera, 0, 1)

        layout.addWidget(label_interface, 1, 0)
        layout.addWidget(self.left_edit_interface, 1, 1)

        layout.addWidget(label_normal, 2, 0)
        layout.addWidget(self.left_edit_normal, 2, 1)

        layout.addWidget(label_step, 3, 0)
        layout.addWidget(self.left_edit_step, 3, 1)

        self.layout.addWidget(pannel, 1, 0)

    def __setup_right(self):
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        window.setWindowTitle('Measuring ROI')

        view = window.addViewBox(row=0, col=0, lockAspect=True)

        #view.setRange(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        #view.setLimits(xMin=0, yMin=0, xMax=image.shape[1], yMax=image.shape[0])
        canvas = pg.ImageItem()
        view.addItem(canvas)

        self.right_canvas = canvas

        pannel = QWidget()
        layout = QGridLayout()
        pannel.setLayout(layout)
        self.btn_load_image = QPushButton('Load Image')
        self.btn_load_camera = QPushButton('Load Camera')

        label_interface  = QLabel('Water Level')
        label_normal  = QLabel('Normal Direction')
        label_step  = QLabel('Step')

        self.right_edit_interface  = QLineEdit('0')
        self.right_edit_normal  = QLineEdit('0, 0, 1')
        self.right_edit_step  = QLineEdit('1')

        layout.addWidget(self.btn_load_image, 0, 0)
        layout.addWidget(self.btn_load_camera, 0, 1)

        layout.addWidget(label_interface, 1, 0)
        layout.addWidget(self.right_edit_interface, 1, 1)

        layout.addWidget(label_normal, 2, 0)
        layout.addWidget(self.right_edit_normal, 2, 1)

        layout.addWidget(label_step, 3, 0)
        layout.addWidget(self.right_edit_step, 3, 1)

        self.layout.addWidget(window, 0, 1)
        self.layout.addWidget(pannel, 1, 1)

def epipolar_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()

if __name__ == "__main__":
    epipolar_app()
