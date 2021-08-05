import os
import re
import sys
import json
import pickle
import base64
import numpy as np
import fish_3d as f3
from PIL import Image
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from io import BytesIO
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QColor, QVector3D, QIcon, QTableWidgetItem
from PyQt5.QtWidgets import QWidget, QMainWindow, QPushButton, QLabel,\
     QGridLayout, QApplication, QHBoxLayout, QLineEdit, QFileDialog, qApp,\
     QDesktopWidget, QMessageBox, QAction, QTableWidget, QAbstractScrollArea,\
     QTableView

"""
Colors that I like
"""
TEAL = QColor(14, 110, 100)
PINK = QColor(254, 176, 191)
GOLD = QColor(254, 208, 9.0)
TOMATO = QColor(249, 82, 60)
CRIMSON = QColor(209, 0, 46)
LIGHTSEAGREEN = QColor(33, 165, 153)
CORNFLOWERBLUE = QColor(82, 127, 232)
LAVENDERBLUSH = QColor(255, 235, 242)

pg.setConfigOptions(imageAxisOrder='row-major')


def warn(message):
    msg = QMessageBox()
    msg.minimumWidth = 300
    msg.setIcon(QMessageBox.Warning)
    msg.setText(message)
    msg.exec_()


class Model():
    """
    self.get_ep_methods: the function to calculate epipolar lines for different views

    Attributes:
        measurements_2d (list): The line segment representing one fish in 3 views,
            each element is a numpy array, shape: (3 [view], 2 [points], 2 [xy])
        measurements_3d (list): The 3D line segment representing one fish,
            each element is a numpy array, shape: (2 [points], 3 [xyz])
    """
    def __init__(self, environment=None):
        self.images = [None, None, None]
        self.cameras = [None, None, None]
        self.env = environment
        self.measurements_2d = []  # "shape": (n, 3-view, 2-points, 2-xy)
        self.measurements_3d = []  # "shape": (n, 2-points, 3-xyz)
        self.measurements_3d_errors = []  # "shape": (n, 2-points,)

    def add_new_measure(self, measure_2d):
        points_3d = np.empty((2, 3))
        errors_3d = np.empty(2)
        # iter the two points, with shape (2-points, 3-view, 2-xy)
        for i, point_v3 in enumerate(np.moveaxis(measure_2d, 1, 0)):
            point_xyz, error = f3.ray_trace.ray_trace_refractive_faster(
                point_v3, self.cameras, self.water_level, self.normal
            )
            points_3d[i] = point_xyz
            errors_3d[i] = error
        self.measurements_2d.append(measure_2d)
        self.measurements_3d.append(points_3d)
        self.measurements_3d_errors.append(errors_3d)

    def remove_measure(self, index):
        for data in (
                self.measurements_2d, self.measurements_3d, self.measurements_3d_errors
        ):
            del(data[index])

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
        """
        default value: 0
        """
        if self.env:
            return float(self.env['z'].text())
        else:
            return 0

    @property
    def normal(self):
        """
        default value: (0, 0, 1)
        """
        if self.env:
            normal_str =  self.env['n'].text()
            normal = re.split(r'[\s,]+', normal_str)
            normal = [int(n) for n in normal]
            return normal
        else:
            return (0, 0, 1)

    @property
    def depth(self):
        """
        default value: 400
        """
        if self.env:
            return float(self.env['depth'].text())
        else:
            return 400

    def export_json(self, filename):
        """
        Export all the data to a json file
        """
        if not self.is_valid:
            raise RuntimeError("Trying to export invalid model")
        if filename[-5:] != ".json":
            filename += ".json"
        image_strings = []
        for im in self.images:
            f = BytesIO()
            np.savez_compressed(f, data=im)
            f.seek(0)
            code = base64.b64encode(f.read())  # bytes
            string = string = code.decode('utf-8')  # str
            image_strings.append(string)
            f.close()
        data = {
            'image_strings': image_strings,
            'cameras': [cam.zip_essential() for cam in self.cameras],
            'measurements_2d': np.array(self.measurements_2d).tolist(),
            'measurements_3d': np.array(self.measurements_3d).tolist(),
            'measurements_3d_errors': np.array(self.measurements_3d_errors).tolist(),
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_json(self, filename):
        """
        Load data from a json file
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        # restoring images
        images = []
        for i, string in enumerate(data['image_strings']):
            code = base64.b64decode(string)
            f = BytesIO(code)
            img = np.load(f, allow_pickle=True)['data']
            f.close()
            images.append(img)
        self.images = images
        # restoring cameras
        cameras = [f3.Camera() for i in range(3)]
        for i, cam in enumerate(cameras):
            cam.unzip_essential(data['cameras'][i])
        self.cameras = cameras
        # restoring measurements
        self.measurements_2d = [np.array(arr) for arr in data['measurements_2d']]
        self.measurements_3d = [np.array(arr) for arr in data['measurements_3d']]
        self.measurements_3d_errors = [
            np.array(arr) for arr in data['measurements_3d_errors']
        ]


class StereoImageItem(pg.ImageItem):
    """
    Canvas for a 3 view setup.

    Attributes:
        model (Model): the model for the calculaion
        index (int): the index of the view
        line (pg.PolyLineROI): the line object for measuring the fish
        plot (pg.ScatterPlotItem): scatter plot for plotting eppipolar line
            of the *self.index* view
        neighbours (StereoImageItem): the other matching views,
    """
    def __init__(self, model, index, epipolar_plots, line, view):
        """
        ImageItem for epipolar representation
        """
        self.model = model
        self.index = index
        self.line = line
        self.view = view
        self.epipolar_plots = epipolar_plots
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
        """
        For the line segment of current view, plot all its corresponding epipolar lines
            on the neighbour views.
        """
        if not self.model.is_valid:
            return
        plot_pens = [
            pg.mkPen(color=TOMATO, width=2),  # color: tomato
            pg.mkPen(color=LIGHTSEAGREEN, width=2),   # color: crimson
        ]

        for ni, neighbour in enumerate(self.neighbours):
            cam_1 = self.model.cameras[self.index]
            idx_2 = self.model.index_map[self.index][ni]
            im_2 = self.model.images[idx_2]
            cam_2 = self.model.cameras[idx_2]
            for pi, (plot, pos) in enumerate(zip(neighbour.epipolar_plots, self.line.listPoints())):
                pos_view = self.line.mapToView(pos)
                x, y = pos_view.x(), pos_view.y()
                epipolar = f3.ray_trace.epipolar_la_draw(
                    [x, y], cam_1, cam_2, im_2,
                    self.model.water_level, self.model.depth, self.model.normal
                )
                if len(epipolar) > 0:
                    plot.setData(
                        x=epipolar.T[0], y=epipolar.T[1], pen=plot_pens[pi],
                    )


class Viewer(QMainWindow):
    def __init__(self, size=(1000, 800)):
        super().__init__()
        self.layout = QGridLayout()
        self.window = QWidget()
        self.env = None
        self.model = None
        self.canvas_items = []
        self.btn_load_image_items = []
        self.btn_load_camera_items = []
        self.measure_plots_2d = []  # shape (n, 3 [view])
        self.measure_plots_3d = []  # shape (n,)
        self.highlight_plot_3d = gl.GLLinePlotItem(
            pos=np.array((np.zeros(3), np.ones(3))),
            color=(254.0/255, 208.0/255, 9.0/255, 1.0),  # gold
            antialias=True, width=8,
        )
        self.highlight_plots_2d = [
            pg.PlotCurveItem(), pg.PlotCurveItem(), pg.PlotCurveItem(),
        ]
        self.size = size
        self.table_closed = True
        self.__setup()

    def __setup(self):
        self.window.setLayout(self.layout)
        self.setCentralWidget(self.window)
        self.__setup_pannel()  # initialise self.env
        self.model = Model(self.env)
        self.__setup_canvas_2d(index=0, row=0, col=0)
        self.__setup_canvas_2d(index=1, row=0, col=1)
        self.__setup_canvas_2d(index=2, row=1, col=0)
        self.__setup_canvas_3d()
        self.__setup_menu()
        self.__setup_table()
        self.__setup_control()
        for canvas in self.canvas_items:
            canvas.add_neighbour(self.canvas_items)

        self.resize(*self.size)
        # move window to the central screen
        qt_rectangle = self.frameGeometry()
        centre_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(centre_point)
        self.move(qt_rectangle.topLeft())

        # status bar
        self.statusBar()

        self.show()

    def __setup_control(self):
        self.btn_load_image_items[0].clicked.connect(lambda x: self.__load_image(0))
        self.btn_load_image_items[1].clicked.connect(lambda x: self.__load_image(1))
        self.btn_load_image_items[2].clicked.connect(lambda x: self.__load_image(2))
        self.btn_load_camera_items[0].clicked.connect(lambda x: self.__load_camera(0))
        self.btn_load_camera_items[1].clicked.connect(lambda x: self.__load_camera(1))
        self.btn_load_camera_items[2].clicked.connect(lambda x: self.__load_camera(2))
        self.btn_measure.clicked.connect(self.measure)
        self.action_measure.triggered.connect(self.measure)
        self.btn_delete.clicked.connect(self.delete)
        self.action_delete.triggered.connect(self.delete)
        self.action_exit.triggered.connect(qApp.quit)
        self.action_export.triggered.connect(self.export)
        self.action_save.triggered.connect(self.save)
        self.action_load.triggered.connect(self.load)
        self.action_env.triggered.connect(self.__show_pannel)
        self.action_table.triggered.connect(self.__show_table)
        self.table.cellClicked.connect(self.select)
        self.table.currentCellChanged.connect(self.select)

    def __setup_canvas_2d(self, index, row, col):
        pannel = QWidget()
        layout = QGridLayout()
        window = pg.GraphicsLayoutWidget()

        btn_load_image = QPushButton('Load Image')
        btn_load_camera = QPushButton('Load Camera')
        epipolar_plots = pg.PlotCurveItem(), pg.PlotCurveItem()

        # setup the line segment measurement tool
        # making sure the color of the line match the color of the epipolar lines
        line = pg.LineSegmentROI(
            [(0,0), (0, 100)],
            pen=pg.mkPen(PINK, width=3),
            hoverPen=pg.mkPen(LAVENDERBLUSH, width=2),
        )
        line.handles[0]['item'].pen = pg.mkPen(TOMATO, width=5)
        line.handles[1]['item'].pen = pg.mkPen(LIGHTSEAGREEN, width=5)
        for handle in line.endpoints:
            handle.currentPen = handle.pen
        line.hide()

        # set the view object to contain the widgets
        view = window.addViewBox(row=0, col=0, lockAspect=True)

        # setup ccanvas
        canvas = StereoImageItem(
            self.model, index=index,
            epipolar_plots=epipolar_plots, line=line, view=view
        )
        canvas.setZValue(-100)

        view.addItem(canvas)
        for plot in epipolar_plots:
            view.addItem(plot)
        view.addItem(line)

        # hack the clicking
        view.mouseClickEvent = canvas.mouseClickEvent

        # place the widgets
        layout.addWidget(window, 0, 0, 1, 2)
        layout.addWidget(btn_load_camera, 1, 0)
        layout.addWidget(btn_load_image, 1, 1)

        # adjust the margin / padding
        layout.setContentsMargins(0, 0, 0, 0)
        window.setContentsMargins(0, 0, 0, 0)
        view.setContentsMargins(0, 0, 0, 0)
        window.setRange(padding=0)

        pannel.setLayout(layout)
        self.layout.addWidget(pannel, row, col)

        self.canvas_items.append(canvas)
        self.btn_load_image_items.append(btn_load_image)
        self.btn_load_camera_items.append(btn_load_camera)

        indices = ['1st', '2nd', '3rd']
        btn_load_image.setStatusTip(f"Load the image from the {indices[index]} view.")
        btn_load_camera.setStatusTip(f"Load the camera from the {indices[index]} view.")
        window.setStatusTip(
            f"The *undistorted* image and measurements of the {indices[index]} view."
        )

    def __setup_canvas_3d(self):
        pannel = QWidget()
        layout = QGridLayout()
        view_3d = gl.GLViewWidget()
        view_3d.opts['distance'] = 80
        self.btn_measure = QPushButton('Measure 3D Line')
        self.btn_measure.setStatusTip(
            "Confirm matched line segments and measure the corresponding 3D line segment."
        )
        self.btn_delete = QPushButton('Delete Measurement')
        self.btn_delete.setStatusTip(
            "Deleting selected 3D line segments from the result table."
        )
        view_3d.setStatusTip("View of the 3D line segments.")

        # place the widgets
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view_3d, 0, 0, 1, 2)
        layout.addWidget(self.btn_measure, 1, 0)
        layout.addWidget(self.btn_delete, 1, 1)
        pannel.setLayout(layout)

        self.layout.addWidget(pannel, 1, 1)
        self.canvas_3d = view_3d

    def __setup_pannel(self):
        """
        the setting pannel
        """
        pannel = QWidget(parent=self.parent())
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
        env = {'z': self.edit_interface, 'n': self.edit_normal, 'depth': self.edit_depth}
        self.env = env
        self.pannel = pannel

    def __setup_table(self):
        table = QTableWidget(0, 2)
        table.setSelectionBehavior(QTableView.SelectRows)
        table.setHorizontalHeaderLabels([f"{'Point 1':^50}", f"{'Point 2':^50}"])
        table.setSizeAdjustPolicy(
            QAbstractScrollArea.AdjustToContents
        )
        table.closeEvent = self.__close_table
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table = table

    def __setup_menu(self):
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)  # for osx only

        # setting up file menu
        file_menu = menu_bar.addMenu('&File')
        save = file_menu.addAction("Save Project")
        save.setStatusTip("Save all the 2D measurements to a json file.")
        load = file_menu.addAction("Load Project")
        load.setStatusTip(
            "Load all the 2D measurements from a json file and calculate 3D line segments."
        )
        export = file_menu.addAction("Export 3D Results")
        export.setStatusTip("Export all the 3D line segments to a csv file.")
        save.setShortcut('Ctrl+S')
        load.setShortcut('Ctrl+L')
        export.setShortcut('Ctrl+Shift+S')
        file_menu.addSeparator()
        exit = file_menu.addAction("Exit")
        exit.setShortcut('Ctrl+Q')
        self.action_exit = exit
        self.action_export = export
        self.action_save = save
        self.action_load = load

        # setting up view menu
        view_menu = menu_bar.addMenu('&Edit')

        measure = view_menu.addAction("Measure")
        measure.setStatusTip("Measure the 3D line segment")
        measure.setShortcut('Ctrl+M')
        self.action_measure = measure

        delete = view_menu.addAction("Delete")
        delete.setStatusTip("Delete the selected line segment")
        delete.setShortcut('Ctrl+D')
        self.action_delete = delete

        table = view_menu.addAction("Result Table")
        table.setStatusTip("Editting the Measurement Result")
        table.setShortcut('Ctrl+T')
        self.action_table = table

        environment = view_menu.addAction("Environment")
        environment.setStatusTip("Editting the environmental variables.")
        environment.setShortcut('Ctrl+E')
        self.action_env = environment

    def __load_image(self, index):
        camera = self.model.cameras[index]
        if isinstance(camera, type(None)):
            warn("Please Load the Corresponding Camera First")
            return
        image_name, _ = QFileDialog.getOpenFileName(
                self, "Select the image", "",
                "Image files (*.tiff *.png *.jpeg);;All Files (*)"
                )
        if image_name:
            img = np.array(Image.open(image_name))
            img = self.model.cameras[index].undistort_image(img)
            self.canvas_items[index].setImage(img)
            pos = QtCore.QPoint(img.shape[1]//2, img.shape[0]//2)
            self.canvas_items[index].line.setPos(pos)
            self.canvas_items[index].line.show()
            self.model.images[index] = img

    def __load_camera(self, index):
        camera_name, _ = QFileDialog.getOpenFileName(
                self, "Select the camera", "", "camera files (*.pkl);;"
                )
        if camera_name:
            with open(camera_name, 'rb') as f:
                camera = pickle.load(f)
            self.model.cameras[index] = camera

    def __show_pannel(self):
        self.pannel.hide()
        self.pannel.show()

    def __refresh_table(self):
        self.table.setRowCount(0)  # deletign all the rows
        for row, measure in enumerate(self.model.measurements_3d):
            self.table.insertRow(self.table.rowCount())
            for i in range(2):
                coordinate = [f"{x:^10.1f}" for x in measure[i]]
                self.table.setItem(row, i, QTableWidgetItem(",".join(coordinate)))

    def __show_table(self):
        self.table.hide()
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.__refresh_table()
        self.table.show()
        self.table_closed = False

    def __close_table(self, event):
        if self.highlight_plot_3d in self.canvas_3d.items:
            self.canvas_3d.removeItem(self.highlight_plot_3d)
        for view_id, canvas in enumerate(self.canvas_items):
            if self.highlight_plots_2d[view_id] in canvas.view.addedItems:
                canvas.view.removeItem(self.highlight_plots_2d[view_id])
        self.table_closed = True
        self.table.hide()

    def __collect_2d_lines(self):
        """
        Appending the line measurement to self.model.segmetns_2d
        """
        measure_2d = np.empty((3, 2, 2))
        for i, canvas in enumerate(self.canvas_items):
            for j, point in enumerate(canvas.line.listPoints()):
                xy = canvas.line.mapToView(point)
                measure_2d[i, j, 0] =xy.x()
                measure_2d[i, j, 1] =xy.y()
        return measure_2d

    def __refresh_measure_plots(self):
        """
        Plotting all the line measurement data from `self.model.measurements_2d`
            on the canvas of different views
        """
        for plots_v3 in self.measure_plots_2d:
            for i, plot in enumerate(plots_v3):
                self.canvas_items[i].view.removeItem(plot)
        self.measure_plots_2d.clear()
        for measure_2d_v3 in self.model.measurements_2d:
            plots_v3 = []
            for i, measure_2d in enumerate(measure_2d_v3):
                new_plot = pg.PlotCurveItem()
                new_plot.setData(
                    *measure_2d.T,
                    pen=pg.mkPen(color=CORNFLOWERBLUE, width=4)
                )
                self.canvas_items[i].view.addItem(new_plot)
                plots_v3.append(new_plot)
            self.measure_plots_2d.append(plots_v3)

    def __refresh_3d_plot(self):
        """
        Plotting all the line measurement data from `self.model.measurements_2d`
            on the canvas of different views
        """
        for item in self.measure_plots_3d:
            self.canvas_3d.removeItem(item)
        self.measure_plots_3d.clear()
        all_points = np.concatenate(self.model.measurements_3d)
        self.canvas_3d.opts['center'] = QVector3D(*all_points.mean(axis=0))
        for line_3d in self.model.measurements_3d:
            plt = gl.GLLinePlotItem(
                pos=line_3d,
                color=(33.0/255, 165.0/255, 153.0/255, 1.0),  # lightseagreen
                antialias=True,
                width=4,
            )
            self.canvas_3d.addItem(plt)
            self.measure_plots_3d.append(plt)

    def __check_and_get_save_name(self, suffix, caption=""):
        """
        Check the existance of valid result and get a save name

        Args:
            suffix (str): the extension of the file (e.g. csv, json).
            caption (str): the caption on the file dialog
        """
        if not self.model.is_valid:
            warn("Please Generate Measurements First")
            return
        if len(self.model.measurements_3d) == 0:
            warn("Please Generate Measurements First")
            return
        save_name, _ = QFileDialog.getSaveFileName(
            caption="Save 3D line measurements",
            filter=f"{suffix.upper()} files (*.{suffix})"
        )
        return save_name

    def measure(self):
        """
        Adding a permenant line plots for the line segments from each view,
        Calculate the 3D lines based on the 2D line segments.
        Renderling the 3D lines in the 3D view.
        """
        if self.model.is_valid:
            measure_2d = self.__collect_2d_lines()
            self.model.add_new_measure(measure_2d)
            self.__refresh_measure_plots()
            self.__refresh_3d_plot()
        else:
            warn("Please load the images and cameras")

    def select(self):
        index = self.table.currentRow()
        self.highlight_plot_3d.setData(
            pos=self.model.measurements_3d[index]
        )
        if self.highlight_plot_3d not in self.canvas_3d.items:
            self.canvas_3d.addItem(self.highlight_plot_3d)
        for view_id, canvas in enumerate(self.canvas_items):
            plot = self.highlight_plots_2d[view_id]
            plot.setData(
                *self.model.measurements_2d[index][view_id].T,
                pen=pg.mkPen(color=GOLD, width=4)
            )
            if plot not in canvas.view.addedItems:
                canvas.view.addItem(self.highlight_plots_2d[view_id])

    def delete(self):
        if self.table_closed:
            warn("Please select a line from the table view")
            return
        else:
            index = self.table.currentRow()
            self.model.remove_measure(index)
            self.__refresh_measure_plots()
            self.__refresh_3d_plot()
            self.__close_table(None)
            self.__show_table()

    def save(self):
        save_name = self.__check_and_get_save_name(
            'json', caption="Save Project as Json"
        )
        if save_name:
            self.model.export_json(save_name)
            self.setWindowTitle(os.path.basename(save_name))

    def load(self):
        filename, _ = QFileDialog.getOpenFileName(
                self, "Select the Project File", "", "json files (*.json);;"
                )
        self.setWindowTitle(os.path.basename(filename))
        if filename:
            self.model.load_json(filename)
            for i, img in enumerate(self.model.images):
                pos = QtCore.QPoint(img.shape[1]//2, img.shape[0]//2)
                self.canvas_items[i].line.setPos(pos)
                self.canvas_items[i].line.show()
                self.canvas_items[i].setImage(img)
            self.__refresh_measure_plots()
            self.__refresh_3d_plot()

    def export(self):
        save_name = self.__check_and_get_save_name(
            'csv', caption="Save 3D line measurements to CSV"
        )
        if save_name:
            data = np.array(self.model.measurements_3d)  # (n, 2, 3)
            error = np.array(self.model.measurements_3d_errors)[:, :, np.newaxis]  # (n, 2, 1)
            data = np.concatenate((data, error), axis=-1)  # (n, 2, 4)
            n = data.shape[0]
            data = data.reshape((n, 8), order='C')  # last axis index changing fastest
            header = "X1,Y1,Z1,Error1,X2,Y2,Z2,Error2"
            np.savetxt(
                save_name, data, delimiter=',', header=header, comments=""
            )


def line_measure_app():
    """
    return the position and size of the ROI
    """
    app = QApplication(sys.argv)
    epp = Viewer()
    app.exec_()


if __name__ == "__main__":
    line_measure_app()
