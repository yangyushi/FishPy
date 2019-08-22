import os
import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QMessageBox,\
                            QGridLayout, QApplication, QHBoxLayout, QFileDialog
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(imageAxisOrder='row-major')

def warn(message):
    msg = QMessageBox()
    msg.minimumWidth = 300
    msg.setIcon(QMessageBox.Warning)
    msg.setText(message)
    msg.exec_()


class Model():
    def __init__(self, images, batch_size=36):
        self.images = images  # numpy.ndarray containing all shape images
        self.labels = np.zeros(self.images.shape[0], np.uint8)  # 0 -> good fish; 1 -> bad fish
        self.last_batch_index = len(images) // batch_size
        self.last_batch_size = len(images) % batch_size
        self.batch_size = batch_size
        self.batch_index = 0
        if len(images) < self.batch_size:
            self.batch = self.images[:len(images)]
        else:
            self.batch = self.images[:self.batch_size]

    @property
    def cursor(self):
        return self.batch_index * self.batch_size

    def mark(self, number):
        index = self.cursor + number
        self.labels[index] = 1

    def next(self):
        if self.batch_index < self.last_batch_index:
            self.batch_index += 1
            self.batch = self.images[self.cursor : self.cursor + self.batch_size]

    def back(self):
        if self.batch_index > 0:
            self.batch_index -= 1
            self.batch = self.images[self.cursor : self.cursor + self.batch_size]


class Photo(pg.ImageItem):
    def __init__(self, index, mother):
        self.index = index
        self.mother = mother
        self.model = mother.model
        self.white = pg.mkPen(color=(255, 255, 255), width=2)
        self.tomato = pg.mkPen(color=(249, 82, 60), width=4)
        pg.ImageItem.__init__(self, border=self.white)
        self.update_color()

    @property
    def label(self):
        index = self.model.cursor + self.index
        return self.model.labels[index]

    def update_color(self):
        self.model = self.mother.model
        if self.label == 0:
            self.setBorder(self.white)
        else:
            self.setBorder(self.tomato)

    def mouseClickEvent(self, event):
        index = self.model.cursor + self.index
        self.model.labels[index] = (self.label + 1) % 2
        self.update_color()


class Viewer(QMainWindow):
    def __init__(self, images=None):
        super().__init__()
        if isinstance(images, type(None)):
            images = np.zeros((36, 10, 10))
        self.model = Model(images)
        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()

    def __setup(self):
        self.setAcceptDrops(True)
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_canvas()
        self.__setup_control()
        self.show()

    def __setup_canvas(self):
        batch = self.model.batch
        image = batch[0]

        window = pg.GraphicsLayoutWidget(border=True)
        window.setWindowTitle('Fish Gallery')
        self.canvases = []

        for row in range(6):
            for col in range(6):
                view = window.addViewBox(row=row, col=col, lockAspect=True)
                index = row * 6 + col
                canvas = Photo(index, self)
                canvas.setImage(batch[index], autoLevels=True)
                view.addItem(canvas)
                self.canvases.append(canvas)
        self.layout.addWidget(window, 0, 0)

    def __setup_control(self):
        pannel = QWidget()
        layout = QHBoxLayout()
        pannel.setLayout(layout)
        self.btn_next = QPushButton('Next')
        self.btn_back = QPushButton('Back')
        self.btn_save = QPushButton('Save')
        self.btn_exit = QPushButton('Exit')
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_back)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_exit)

        self.btn_next.clicked.connect(self.next)
        self.btn_back.clicked.connect(self.back)
        self.btn_save.clicked.connect(self.save)
        self.btn_exit.clicked.connect(self.close)

        self.layout.addWidget(pannel, 1, 0)

    def keyPressEvent(self, event):
        if event.key() in [Qt.Key_Right, Qt.Key_Down, Qt.Key_D, Qt.Key_L, Qt.Key_J]:
            self.next()
        elif event.key() in [Qt.Key_Left, Qt.Key_Up, Qt.Key_A, Qt.Key_H, Qt.Key_K]:
            self.back()
        elif event.key() in [Qt.Key_S, Qt.Key_Return, Qt.Key_Space]:
            self.save()
        elif event.key() == Qt.Key_Escape:
            self.close()

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        filename = event.mimeData().text()
        if os.name == 'nt':
            filename = filename[8:]
        elif os.name == 'posix':
            filename = filename[7:-2]
        else:
            filename = filename[7:]
        if '\n' in filename:
            warn("I can't accept multiple files")
            return 
        if 'npy' not in filename:
            warn("Only *.npy file is accepted!")
            return
        data = np.load(filename)
        if data.ndim == 3:
            self.model = Model(data)
            self.update()
        elif (data.ndim == 1) and (len(data) == len(self.model.images)):
            if (data.max() == 1) and (data.min() == 0):
                self.model.labels = data
                self.update()
            else:
                warn("labels have the wrong value")
        else:
            warn("Please give me images (n, x, y,) or binary labels (x,)")

    def update(self):
        for row in range(6):
            for col in range(6):
                index = row * 6 + col
                if index < len(self.model.batch):
                    self.canvases[index].setImage(self.model.batch[index])
                    self.canvases[index].update_color()
                else:
                    size = self.model.images[0].shape
                    self.canvases[index].setImage(np.zeros(size, dtype=np.uint8))

    def next(self):
        self.model.next()
        self.update()

    def back(self):
        self.model.back()
        self.update()

    def save(self):
        file_name, _ = QFileDialog.getSaveFileName(
                self, "Save the labels", "label_", "(*.npy);;"
                )
        np.save(file_name, self.model.labels)


def shape_gallery():
    app = QApplication(sys.argv)
    measure = Viewer()
    app.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    measure = Viewer()
    app.exec_()
