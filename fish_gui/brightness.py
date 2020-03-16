#!/usr/bin/env python3
import json
import sys
from tplight import LB130
from PyQt5.QtWidgets import QWidget, QApplication,\
    QMainWindow, QLabel, QSlider, QHBoxLayout
from PyQt5.QtCore import Qt


class Controller(QMainWindow):
    def __init__(self, ip="192.168.0.1"):
        super().__init__()
        self.light = LB130(ip)
        self.light.transition_period = 0
        self.window = QWidget()
        self.__setup()

    def brightness(self):
        state = json.loads(self.light.status())
        return int(state['system']['get_sysinfo']['light_state']['brightness'])

    def __setup(self):
        self.layout = QHBoxLayout()
        self.setCentralWidget(self.window)
        self.window.setLayout(self.layout)
        self.slide = QSlider(Qt.Horizontal, self)
        self.value = QLabel("0")
        self.value.setText(str(self.brightness()))
        self.layout.addWidget(QLabel("Brightness"))
        self.layout.addWidget(self.slide)
        self.layout.addWidget(self.value)
        self.slide.valueChanged.connect(self.__change_brightness)
        self.slide.sliderReleased.connect(self.__set_brightness)
        self.show()

    def __change_brightness(self, value):
        if value > 100:
            value = 100
        elif value < 0:
            value = 0
        value = int(value)
        self.value.setText(str(value))

    def __set_brightness(self):
        self.light.brightness = int(self.value.text())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    control = Controller()
    app.exec_()
