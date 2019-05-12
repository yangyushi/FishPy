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
        normal = re.split(r'[\s,]+', normal_str)
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

    def get_ep21(self, uv):
        if self.is_valid:
            return f3.ray_trace.epipolar_draw(
                    uv, self.camera_2, self.camera_1, self.image_1, self.water_level, self.depth, self.normal
                    )


class StereoImageItem(pg.ImageItem):
    def __init__(self, model):
        """
        ImageItem for epipolar representation
        """
        self.model = model
        self.buddy = None
        pg.ImageItem.__init__(self)

    def accept_buddy(self, buddy):
        self.buddy = buddy

    def mouseClickEvent(self, event):
        print('test click')

class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_pannel()
        self.model = Model(self.env)
        self.__setup_left()
        self.__setup_right()
        self.left_canvas.accept_buddy(self.right_canvas)
        self.right_canvas.accept_buddy(self.left_canvas)
        self.show()

    def __setup_left(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        view = window.addViewBox(row=0, col=0, lockAspect=True)

        self.btn_load_image_left = QPushButton('Load Image')
        self.btn_load_camera_left = QPushButton('Load Camera')
        canvas = StereoImageItem(self.model)

        view.addItem(canvas)
        self.left_canvas = canvas

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image_left, 1, 0)
        layout.addWidget(self.btn_load_camera_left, 1, 1)

        pannel.setLayout(layout)
        self.layout.addWidget(pannel, 0, 0)

    def __setup_right(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        canvas = StereoImageItem(self.model)

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        view.addItem(canvas)
        self.right_canvas = canvas
        self.btn_load_image_right = QPushButton('Load Image')
        self.btn_load_camera_right = QPushButton('Load Camera')

        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image_right, 1, 0)
        layout.addWidget(self.btn_load_camera_right, 1, 1)

        pannel.setLayout(layout)
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
        env = {'z': self.edit_interface, 'n': self.edit_normal, 'depth': self.edit_depth}
        self.env = env

    def __get_left_uv(self, event):
        """store the x, y positions to self.xy_current"""
        scene_pos = event.scenePos()
        plot_pos = self.left_canvas.vb.mapSceneToView(scene_pos)
        u, v = plot_pos.x(), plot_pos.y()
        print(u, v)
        return u, v


def epipolar_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()

if __name__ == "__main__":
    epipolar_app()
