import sys
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel, QGridLayout, QApplication, QHBoxLayout, QSlider, QFileDialog
import numpy as np
from matplotlib import cm
from scipy import ndimage
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import fish_track as ft

pg.setConfigOptions(imageAxisOrder='row-major')

def get_fg_color(fg):
    rgba = np.ones((fg.shape[0], fg.shape[1], 4))
    # color "tomato" in matplotlib 
    rgba[:, :, 0] = 252/255
    rgba[:, :, 1] = 100/255
    rgba[:, :, 2] = 78/255
    rgba *= np.expand_dims(fg > 0, -1)
    return rgba


class Model():
    """showing images at different frames"""
    def __init__(self, images, config_file, background=None):
        self.images = images
        self.background = background
        config = ft.utility.Configure(config_file)

        roi = config.Process.roi
        x0, y0, size_x, size_y = [int(x) for x in roi.split(', ')]
        self.roi = (slice(y0, y0 + size_y, None), slice(x0, x0 + size_x))

        if config.Process.gaussian_sigma != 0:
            denoise = lambda x: ndimage.gaussian_filter(x, config.Process.gaussian_sigma)
        else:
            denoise = lambda x: x

        if config.Process.normalise == 'std':
            normalise = lambda x: x / x.std()
        elif config.Process.normalise == 'max':
            normalise = lambda x: x / x.max()
        elif config.Process.normalise == 'mid':
            def normalise(x): return x / np.median(x)
        elif config.Process.normalise == 'mean':
            normalise = lambda x: x / x.mean()
        elif config.Process.normalise == 'None':
            normalise = lambda x: x

        self.process = lambda x: normalise(denoise(x))
        self.threshold = config.Fish.threshold
        self.cursor = -1
        self.history = [self.process(next(images))]
        self.image = self.history[0]

    @property
    def label(self):
        if isinstance(self.background, type(None)):
            fg = self.image.max() - self.image
        else:
            fg = self.background - self.image
            fg -= fg.min()
        fg_binary = fg > (fg[self.roi].max() * self.threshold)
        return get_fg_color(fg_binary)

    @property
    def next(self):
        if self.cursor == len(self.history) - 1:
            try:
                self.image = self.process(next(self.images))
            except StopIteration:
                return self.image, self.label
            self.history.append(self.image)
            self.cursor += 1
            return self.image, self.label
        else:
            self.cursor += 1
            self.image = self.history[self.cursor]
            return self.image, self.label

    @property
    def back(self):
        self.cursor -= 1
        self.cursor = max(self.cursor, 0)
        self.image = self.history[self.cursor]
        return self.image, self.label


class Viewer(QMainWindow):
    def __init__(self, images, config_path, background):
        super().__init__()
        if isinstance(background, type(None)):
            self.model = Model(images, config_path)
        else:
            self.model = Model(images, config_path, np.load(background))
        self.result = self.model.threshold
        self.layout = QGridLayout()
        self.window = QWidget()
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_canvas()
        self.__setup_control()
        self.show()

    def __setup_canvas(self):
        image, label = self.model.next
        window = pg.GraphicsLayoutWidget(border=True)
        window.setWindowTitle('Measuring ROI')


        view = window.addViewBox(row=0, col=0, lockAspect=True)

        view.setRange(QtCore.QRectF(0, 0, image.shape[1], image.shape[0]))
        view.setLimits(xMin=0, yMin=0, xMax=image.shape[1], yMax=image.shape[0])
        canvas = pg.ImageItem()
        canvas_label = pg.ImageItem()

        canvas.setImage(image)
        canvas_label.setImage(label)

        view.addItem(canvas)
        view.addItem(canvas_label)

        self.canvas = canvas
        self.canvas_label = canvas_label
        self.layout.addWidget(window, 0, 0)

    def __setup_control(self):
        pannel = QWidget()
        layout = QGridLayout()
        pannel.setLayout(layout)

        self.label_threshold = QLabel(f"Threshold ({self.model.threshold:.2f})")
        self.sld_threshold = QSlider(QtCore.Qt.Horizontal)
        self.sld_threshold.setValue(self.model.threshold * 100)
        self.btn_bg = QPushButton('Background')
        self.btn_next = QPushButton('Next')
        self.btn_back = QPushButton('Back')
        self.btn_save = QPushButton('Save')
        self.btn_exit = QPushButton('Exit')

        layout.addWidget(self.btn_bg, 0, 0)
        layout.addWidget(self.label_threshold, 0, 1)
        layout.addWidget(self.sld_threshold, 0, 2, 1, 2)
        layout.addWidget(self.btn_next, 1, 0)
        layout.addWidget(self.btn_back, 1, 1)
        layout.addWidget(self.btn_save, 1, 2)
        layout.addWidget(self.btn_exit, 1, 3)

        self.sld_threshold.sliderReleased.connect(self.update)
        self.sld_threshold.valueChanged.connect(self.update_text)
        self.btn_bg.clicked.connect(self.get_background)
        self.btn_next.clicked.connect(self.next)
        self.btn_back.clicked.connect(self.back)
        self.btn_save.clicked.connect(self.save)
        self.btn_exit.clicked.connect(self.close)

        self.layout.addWidget(pannel, 1, 0)

    def get_background(self):
        file_name, _ = QFileDialog.getOpenFileName(
                self, "Select the background file", "", "background files (*.npy);;"
                )
        if not isinstance(file_name, type(None)):
            self.model.background = np.load(file_name)
            self.update()

    def next(self):
        image, label = self.model.next
        self.canvas.setImage(image)
        self.canvas_label.setImage(label)

    def back(self):
        image, label = self.model.back
        self.canvas.setImage(image)
        self.canvas_label.setImage(label)

    def save(self):
        self.result = self.model.threshold
        self.close()

    def update_text(self):
        value = self.sld_threshold.value() / 100
        self.label_threshold.setText(f"Threshold ({value:.2f})")

    def update(self):
        self.model.threshold = self.sld_threshold.value() / 100
        self.label_threshold.setText(f"Threshold ({self.model.threshold:.2f})")
        self.canvas.setImage(self.model.image)
        self.canvas_label.setImage(self.model.label)


def get_threshold(images, config, background=None):
    """
    get a proper threshold value by manually measurement
    """
    app = QApplication(sys.argv)
    measure = Viewer(images, config, background)
    app.exec_()
    return measure.result


if __name__ == "__main__":
    import fish_track as ft
    path = '../test/images/cam-1'
    config = '../script/track/configure.ini'
    bg = '../test/images/background-cam-1.npy'
    images = ft.read.iter_image_sequence(path)
    get_threshold(images, config, bg)
