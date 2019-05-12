import re
import fish_3d as f3
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout, QLineEdit
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')

class Model():
    def __init__(self, environment):
        self.image_1 = None 
        self.image_2 = None 
        self.camera_1 = None 
        self.camera_2 = None 
        self.env = environment

    @property
    def is_valid(self):
        return None not in (self.image_1, self.image_2, self.camera_1, self.camera_2)

    @property
    def water_level(self):
        return float(self.env['z'].text())

    @property
    def normal(self):
        normal_str =  self.env['n'].text()
        normal = re.split(r'[\s,]+', n_str)
        normal = [int(n) for n in normal]
        return normal

    @property
    def depth(self):
        return float(self.env['depth'].text())

    def get_ep12(self, uv):
        if self.is_valid:
            return f3.ray_trace.epipolar_draw(
                    uv, self.camera_1, self.camera_2, self.image_2, self.water_level, self.depth, self.normal
                    )

class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()
        env = {'z': self.edit_interface, 'n': self.edit_normal, 'depth': self.edit_depth}
        self.model = Model(env)

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_left()
        self.__setup_right()
        self.__setup_pannel()
        self.show()

    def __setup_left(self):
        pannel = QWidget()
        layout = QGridLayout()
        pannel.setLayout(layout)
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        window.setWindowTitle('Measuring ROI')
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        canvas = pg.ImageItem()
        view.addItem(canvas)
        self.left_canvas = canvas
        self.btn_load_image = QPushButton('Load Image')
        self.btn_load_camera = QPushButton('Load Camera')
        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image, 1, 0)
        layout.addWidget(self.btn_load_camera, 1, 1)
        self.layout.addWidget(pannel, 0, 0)

    def __setup_right(self):
        pannel = QWidget()
        layout = QGridLayout()
        pannel.setLayout(layout)
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        window.setWindowTitle('Measuring ROI')
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        canvas = pg.ImageItem()
        view.addItem(canvas)
        self.right_canvas = canvas
        self.btn_load_image = QPushButton('Load Image')
        self.btn_load_camera = QPushButton('Load Camera')
        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image, 1, 0)
        layout.addWidget(self.btn_load_camera, 1, 1)
        self.layout.addWidget(pannel, 0, 1)

    def __setup_pannel(self):
        pannel = QWidget()
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.edit_interface = QLineEdit('0')
        self.edit_normal = QLineEdit('0, 0, 1')
        self.edit_depth = QLineEdit('400')
        label_interface  = QLabel('Water Level')
        label_normal  = QLabel('Normal Direction')
        label_step  = QLabel('Water Depth')
        layout.addWidget(label_interface)
        layout.addWidget(self.edit_interface)
        layout.addWidget(label_step)
        layout.addWidget(self.edit_depth)
        layout.addWidget(label_normal)
        layout.addWidget(self.edit_normal)
        self.layout.addWidget(pannel, 1, 0, 1, 2)

def epipolar_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()

if __name__ == "__main__":
    epipolar_app()
