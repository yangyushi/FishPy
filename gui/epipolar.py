import re
import fish_3d as f3
import sys
import pickle
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout,\
                            QApplication, QHBoxLayout, QLineEdit, QFileDialog
from PyQt5.QtGui import QColor
from PIL import Image
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
        for item in (self.image_1, self.image_2, self.camera_1, self.camera_2):
            if isinstance(item, type(None)):
                return False
        return True

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
        else:
            return []

    def get_ep21(self, uv):
        if self.is_valid:
            return f3.ray_trace.epipolar_draw(
                    uv, self.camera_2, self.camera_1, self.image_1, self.water_level, self.depth, self.normal
                    )
        else:
            return []


class StereoImageItem(pg.ImageItem):
    def __init__(self, model, label, plot):
        """
        ImageItem for epipolar representation
        """
        self.model = model
        self.label = label
        self.plot = plot
        self.buddy = None
        pg.ImageItem.__init__(self)

    def accept_buddy(self, buddy):
        self.buddy = buddy

    def mouseClickEvent(self, event):
        edge = pg.mkPen(None)
        fill = pg.mkBrush(color=QColor(249, 82, 60))  # tomato in matplotlib

        pos = event.pos()
        uv = self.mapToView(pos)
        u, v = uv.x(), uv.y()
        self.plot.clear()
        self.buddy.clear()
        self.plot.addPoints(x=[u], y=[v], pen=edge, brush=fill)
        if self.label == 1:
            epipolar_line = self.model.get_ep12([v, u])
        elif self.label == 2:
            epipolar_line = self.model.get_ep21([v, u])
        else:
            raise ValueError("Wrong StereoImageItem Label, ", self.label)
        if len(epipolar_line) > 0:
            self.buddy.addPoints(x=epipolar_line.T[1], y=epipolar_line.T[0], pen=edge, brush=fill)


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
        self.__setup_control()
        self.left_canvas.accept_buddy(self.right_plot)
        self.right_canvas.accept_buddy(self.left_plot)
        self.show()

    def __setup_control(self):
        self.btn_load_image_left.clicked.connect(self.__load_image_left)
        self.btn_load_image_right.clicked.connect(self.__load_image_right)
        self.btn_load_camera_left.clicked.connect(self.__load_camera_left)
        self.btn_load_camera_right.clicked.connect(self.__load_camera_right)

    def __setup_left(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget(show=True, border=True)
        view = window.addViewBox(row=0, col=0, lockAspect=True)

        self.btn_load_image_left = QPushButton('Load Image')
        self.btn_load_camera_left = QPushButton('Load Camera')
        plot = pg.ScatterPlotItem()
        canvas = StereoImageItem(self.model, label=1, plot=plot)
        canvas.setZValue(-100)

        view.addItem(canvas)
        view.addItem(plot)
        self.left_canvas = canvas
        self.left_plot = plot

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
        plot = pg.ScatterPlotItem()
        canvas = StereoImageItem(self.model, label=2, plot=plot)
        canvas.setZValue(-100)

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        view.addItem(canvas)
        view.addItem(plot)
        self.right_plot = plot
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

    def __load_image_left(self):
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "", "tiff images (*.tiff);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            self.left_canvas.setImage(img)
            self.model.image_1 = img

    def __load_image_right(self):
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "tiff images (*.tiff);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            self.right_canvas.setImage(img)
            self.model.image_2 = img

    def __load_camera_left(self):
        """todo: draw the origin on the image"""
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        with open(camera_name, 'rb') as f:
            camera = pickle.load(f)
        self.model.camera_1 = camera

    def __load_camera_right(self):
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        with open(camera_name, 'rb') as f:
            camera = pickle.load(f)
        self.model.camera_2 = camera


def epipolar_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()


if __name__ == "__main__":
    epipolar_app()
