"""
measure a point in an image
"""
from PIL import Image
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout, QFileDialog, QMessageBox
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')


def show_info(message):
    msg = QMessageBox()
    msg.minimumWidth = 300
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    msg.exec_()


class PointMeasureItem(pg.ImageItem):
    def __init__(self, parent, scatter):
        super().__init__(parent=parent)
        self.scatter = scatter

    def mouseClickEvent(self, event):
        pos = event.pos()
        self.scatter.setData([pos.x()], [pos.y()])


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.window = QWidget(parent=self)
        self.scatter = pg.ScatterPlotItem(
            parent=self,
            x=[], y=[], pen=(0,0,0), brush=(255,255,255), symbol='+',
            size=24, pxMode=True, antialias=False
        )
        self.positions = []
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_canvas()
        self.__setup_control()
        self.show()

    def __setup_canvas(self):
        window = pg.GraphicsLayoutWidget(border=True)
        window.setWindowTitle('Measuring Point')
        self.view = window.addViewBox(row=0, col=0, lockAspect=True)
        self.canvas = PointMeasureItem(self, self.scatter)
        self.view.addItem(self.canvas)
        self.view.addItem(self.scatter)
        self.layout.addWidget(window, 0, 0)

    def __load_image(self, image):
        self.view.setRange(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        self.view.setLimits(xMin=0, yMin=0, xMax=image.shape[1], yMax=image.shape[0])
        self.canvas.setImage(image)

    def __setup_control(self):
        pannel = QWidget(self)
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.btn_load = QPushButton('Load Image', parent=self)
        self.btn_save = QPushButton('Save', parent=self)
        self.btn_exit = QPushButton('Exit', parent=self)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_exit)

        self.btn_load.clicked.connect(self.load)
        self.btn_save.clicked.connect(self.save)
        self.btn_exit.clicked.connect(self.close)

        self.layout.addWidget(pannel, 1, 0)

    def load(self):
        file_path, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "image (*.tiff, *.jpg, *.jpeg, *.png);;All Files (*)"
                )

        if file_path:
            image = np.array(Image.open(file_path).convert("L"))
            self.__load_image(image)

    def save(self):
        show_info("Saving new position")
        self.positions.append(self.position)

    @property
    def position(self):
        return np.squeeze(np.array(self.scatter.getData()))


def measure_point():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    measure = Viewer()
    app.exec_()
    return measure.positions

if __name__ == "__main__":
    print(measure_point())
