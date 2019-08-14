import cv2
import sys
import os
from PIL import Image
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout, QLineEdit, QMessageBox
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from typing import List, Tuple

pg.setConfigOptions(imageAxisOrder='row-major')

def warn(message):
    msg = QMessageBox()
    msg.minimumWidth = 300
    msg.setIcon(QMessageBox.Warning)
    msg.setText(message)
    msg.exec_()


class Model():
    def __init__(self, image: np.ndarray, corner_number: Tuple[int]):
        self.image = image
        self.corner_number = corner_number
        self.corners = self.__detect()

    def __detect(self):
        ret, corners = cv2.findChessboardCorners(self.image, self.corner_number,
                flags=sum((
                    cv2.CALIB_CB_FAST_CHECK,
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    )))
        return np.squeeze(corners)


class Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.setAcceptDrops(True)
        self.__setup_canvas()
        self.__setup_control()
        self.show()

    def __setup_canvas(self):
        window = pg.GraphicsLayoutWidget(border=True)
        window.setWindowTitle('Checking Order')
        view = window.addViewBox(row=0, col=0, lockAspect=True)
        canvas = pg.ImageItem()
        pen = pg.mkPen(color=(252, 100, 78), width=2.5)
        plot = pg.PlotDataItem(pen=pen)
        scatter = pg.ScatterPlotItem(brush='w', pen=pen)
        view.addItem(canvas)
        view.addItem(plot)
        view.addItem(scatter)
        self.canvas = canvas
        self.plot = plot
        self.scatter = scatter
        self.layout.addWidget(window, 0, 0)

    def __setup_control(self):
        pannel = QWidget()
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.label_n1 = QLabel('Column Number')
        self.edit_n1 = QLineEdit('23')
        self.label_n2 = QLabel('Row Number ')
        self.edit_n2 = QLineEdit('15')
        layout.addWidget(self.label_n1)
        layout.addWidget(self.edit_n1)
        layout.addWidget(self.label_n2)
        layout.addWidget(self.edit_n2)

        self.layout.addWidget(pannel, 1, 0)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        filename = event.mimeData().text()  # get the filename
        if os.name == 'nt':
            filename = filename[8:]
        else:
            filename = filename[7:]
        if '\n' in filename:
            warn('Can not read multiple files')
            return None
        try:
            image = np.array(Image.open(filename).convert('L'))
        except OSError:
            warn("This image format is not supported")
            return
        try:
            corner_number = (
                int(self.edit_n1.text()),
                int(self.edit_n2.text())
            )
        except ValueError:
            warn("Please input numbers")
            return
        self.model = Model(image, corner_number)
        self.canvas.setImage(image)
        self.plot.setData(self.model.corners)
        init_pos = np.vstack(self.model.corners[0])
        self.scatter.setData(*init_pos)


def check_order():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    checker = Viewer()
    app.exec_()

if __name__ == "__main__":
    check_order()
