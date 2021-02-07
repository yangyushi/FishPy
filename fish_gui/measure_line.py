import re
import fish_3d as f3
import sys
import pickle
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel,\
     QGridLayout, QApplication, QHBoxLayout, QLineEdit, QFileDialog
from PyQt5.QtGui import QColor
from PIL import Image
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')


class Model():
    """
    self.get_ep_methods: the function to calculate epipolar lines for different views
    """
    def __init__(self, environment):
        self.images = [None, None, None]
        self.cameras = [None, None, None]
        self.env = environment

    @property
    def index_map(self):
        """
        index_map[index_v1][i] = ith neighbour

        ..code-block ::

           index_v1   |  first neighbour  | second neighbour
          ---------------------------------------------------
            0         |  1                |  2
            1         |  0                |  2
            2         |  0                |  1

        """
        index_map = {}
        for v1 in range(3):
            index_map[v1] = []
            for v2 in range(3):
                if v1 != v2:
                    index_map[v1].append(v2)
        return index_map

    @property
    def is_valid(self):
        for item in (self.images):
            if isinstance(item, type(None)):
                return False
        for item in (self.cameras):
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


class StereoImageItem(pg.ImageItem):
    """
    Canvas for a 3 view setup.

    self.model (Model): the model for the calculaion
    self.index (int): the index of the view
    self.line (pg.PolyLineROI): the line object for measuring the fish
    self.plot (pg.ScatterPlotItem): scatter plot for plotting eppipolar line
        of the *self.index* view
    self.neighbours (StereoImageItem): the other matching views,
    """
    def __init__(self, model, label, plot, line):
        """
        ImageItem for epipolar representation
        """
        self.model = model
        self.index = label
        self.line = line
        self.plot = plot
        self.neighbours = [None, None]
        pg.ImageItem.__init__(self)

    def add_neighbour(self, stereo_images):
        """
        Adding the neighbour views with a fix order order.
        For different view, the order is different.

        ..code-block ::

          self.index  | neighbour_1.index | neighbour_2.index
          ---------------------------------------------------
            0         |  1                |  2
            1         |  0                |  2
            2         |  0                |  1

        Args:
            stereo_images (tuple): a collection of StereoImageItem objects

        Return:
            None
        """
        neighbour_indices = [si.index for si in stereo_images]
        self.neighbours = []
        for i in range(3):
            if i != self.index:
                si = neighbour_indices.index(i)
                self.neighbours.append(stereo_images[si])

    def mouseClickEvent(self, event):
        if not self.model.is_valid:
            return
        edge = pg.mkPen(None)
        fill = pg.mkBrush(color=QColor(249, 82, 60))  # tomato in matplotlib

        pos = event.pos()
        xy = self.mapToView(pos)
        x, y = xy.x(), xy.y()
        self.plot.clear()
        self.plot.addPoints(x=[x], y=[y], pen=edge, brush=fill)
        for ni, neighbour in enumerate(self.neighbours):
            neighbour.plot.clear()
            cam_1 = self.model.cameras[self.index]
            idx_2 = self.model.index_map[self.index][ni]
            im_2 = self.model.images[idx_2]
            cam_2 = self.model.cameras[idx_2]
            epipolar = f3.ray_trace.epipolar_la_draw(
                [x, y], cam_1, cam_2, im_2,
                self.model.water_level, self.model.depth, self.model.normal
            )
            if len(epipolar) > 0:
                neighbour.plot.addPoints(
                    x=epipolar.T[0], y=epipolar.T[1], pen=edge, brush=fill
                )


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.window = QWidget()
        self.env = None
        self.model = None
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_pannel()  # initialise self.env
        self.model = Model(self.env)
        self.__setup_v1()
        self.__setup_v2()
        self.__setup_v3()
        self.__setup_control()
        canvases = [self.canvas_v1, self.canvas_v2, self.canvas_v3]
        self.canvas_v1.add_neighbour(canvases)
        self.canvas_v2.add_neighbour(canvases)
        self.canvas_v3.add_neighbour(canvases)
        self.show()

    def __setup_control(self):
        self.btn_load_image_v1.clicked.connect(self.__load_image_v1)
        self.btn_load_image_v2.clicked.connect(self.__load_image_v2)
        self.btn_load_image_v3.clicked.connect(self.__load_image_v3)
        self.btn_load_camera_v1.clicked.connect(self.__load_camera_v1)
        self.btn_load_camera_v2.clicked.connect(self.__load_camera_v2)
        self.btn_load_camera_v3.clicked.connect(self.__load_camera_v3)

    def __setup_v1(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget()
        view = window.addViewBox(row=0, col=0, lockAspect=True)

        self.btn_load_image_v1 = QPushButton('Load Image')
        self.btn_load_camera_v1 = QPushButton('Load Camera')
        plot = pg.ScatterPlotItem()
        line = pg.PolyLineROI([(0,0), (10, 10)])

        # setup ccanvas
        canvas = StereoImageItem(self.model, label=0, plot=plot, line=line)
        canvas.setZValue(-100)

        view.addItem(canvas)
        view.addItem(plot)
        view.addItem(line)
        self.canvas_v1 = canvas
        self.plot_v1 = plot

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image_v1, 1, 0)
        layout.addWidget(self.btn_load_camera_v1, 1, 1)

        pannel.setLayout(layout)
        self.layout.addWidget(pannel, 0, 0)

    def __setup_v2(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget()
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        plot = pg.ScatterPlotItem()
        line = pg.PolyLineROI([(0,0), (10, 10)])

        # setup canvas
        canvas = StereoImageItem(self.model, label=1, plot=plot, line=line)
        canvas.setZValue(-100)

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        view.addItem(canvas)
        view.addItem(plot)
        view.addItem(line)
        self.plot_v2 = plot
        self.canvas_v2 = canvas
        self.btn_load_image_v2 = QPushButton('Load Image')
        self.btn_load_camera_v2 = QPushButton('Load Camera')

        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image_v2, 1, 0)
        layout.addWidget(self.btn_load_camera_v2, 1, 1)

        pannel.setLayout(layout)
        self.layout.addWidget(pannel, 0, 1)

    def __setup_v3(self):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget()
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        plot = pg.ScatterPlotItem()
        line = pg.PolyLineROI([(0,0), (10, 10)])

        # setup canvas
        canvas = StereoImageItem(self.model, label=2, plot=plot, line=line)
        canvas.setZValue(-100)

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent
        view.addItem(canvas)
        view.addItem(plot)
        view.addItem(line)
        self.plot_v3 = plot
        self.canvas_v3 = canvas
        self.btn_load_image_v3 = QPushButton('Load Image')
        self.btn_load_camera_v3 = QPushButton('Load Camera')

        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(self.btn_load_image_v3, 1, 0)
        layout.addWidget(self.btn_load_camera_v3, 1, 1)

        pannel.setLayout(layout)
        self.layout.addWidget(pannel, 1, 0)

    def __setup_pannel(self):
        pannel = QWidget()
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.edit_interface = QLineEdit('0')
        self.edit_normal = QLineEdit('0, 0, 1')
        self.edit_depth = QLineEdit('400')
        self.btn_save_2d = QPushButton('Save 2D Results')
        self.btn_save_3d = QPushButton('Save 3D Results')
        label_interface  = QLabel('Water Level')
        label_normal  = QLabel('Normal Direction')
        label_step  = QLabel('Water Depth')
        layout.addWidget(self.btn_save_2d)
        layout.addWidget(self.btn_save_3d)
        layout.addWidget(label_interface)
        layout.addWidget(self.edit_interface)
        layout.addWidget(label_step)
        layout.addWidget(self.edit_depth)
        layout.addWidget(label_normal)
        layout.addWidget(self.edit_normal)
        self.layout.addWidget(pannel, 2, 0, 1, 2)
        env = {'z': self.edit_interface, 'n': self.edit_normal, 'depth': self.edit_depth}
        self.env = env

    def __load_image_v1(self):
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "Image files (*.tiff *.png *.jpeg);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            self.canvas_v1.setImage(img)
            self.model.images[0] = img

    def __load_image_v2(self):
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "Image files (*.tiff *.png *.jpeg);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            self.canvas_v2.setImage(img)
            self.model.images[1] = img

    def __load_image_v3(self):
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "Image files (*.tiff *.png *.jpeg);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            self.canvas_v3.setImage(img)
            self.model.images[2] = img

    def __load_camera_v1(self):
        """todo: draw the origin on the image"""
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        with open(camera_name, 'rb') as f:
            camera = pickle.load(f)
        self.model.cameras[0] = camera

    def __load_camera_v2(self):
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        with open(camera_name, 'rb') as f:
            camera = pickle.load(f)
        self.model.cameras[1] = camera

    def __load_camera_v3(self):
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        with open(camera_name, 'rb') as f:
            camera = pickle.load(f)
        self.model.cameras[2] = camera


def epipolar_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()


if __name__ == "__main__":
    epipolar_app()
