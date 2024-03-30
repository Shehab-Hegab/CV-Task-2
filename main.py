import sys

import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from scipy import ndimage
import numpy as np
from PIL import Image

def canny_edge_detector(image, low_threshold, high_threshold):
    smoothed_image = ndimage.gaussian_filter(image, 1)

    gradient_x = ndimage.sobel(smoothed_image, axis=0)
    gradient_y = ndimage.sobel(smoothed_image, axis=1)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    edge_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            angle = gradient_direction[i, j]
            if (0 <= angle < np.pi / 4) or (7 * np.pi / 4 <= angle <= 2 * np.pi):
                neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
            elif (np.pi / 4 <= angle < 3 * np.pi / 4):
                neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
            elif (3 * np.pi / 4 <= angle < 5 * np.pi / 4):
                neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
            else:
                neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
            if gradient_magnitude[i, j] >= max(neighbors):
                edge_image[i, j] = gradient_magnitude[i, j]

    strong_edges = edge_image > high_threshold
    weak_edges = (edge_image >= low_threshold) & (edge_image <= high_threshold)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if weak_edges[i, j]:
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    edge_image[i, j] = 255
                else:
                    edge_image[i, j] = 0

    return edge_image

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("mainwindow.ui", self)  # Load the UI file

        self.LowThresholdSlider.setMinimum(0)
        self.LowThresholdSlider.setMaximum(255)
        self.HighThresholdSlider.setMinimum(0)
        self.HighThresholdSlider.setMaximum(255)

        self.LowThresholdSlider.setValue(50)
        self.HighThresholdSlider.setValue(100)

        self.pushButton_equalize_1.clicked.connect(self.load_image)
        self.comboBox.currentIndexChanged.connect(self.detect_edges)
        self.LowThresholdSlider.sliderReleased.connect(self.detect_edges)
        self.HighThresholdSlider.sliderReleased.connect(self.detect_edges)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                self.image = cv2.resize(image, (300, 300))  # Resize the image to reduce processing time
                self.display_image(self.image, self.label_equalize_input_3)
                self.detect_edges()  # Automatically detect edges after loading image

    def detect_edges(self):
        if hasattr(self, 'image') and self.image is not None:
            low_threshold = self.LowThresholdSlider.value()
            high_threshold = self.HighThresholdSlider.value()
            edges = cv2.Canny(self.image, low_threshold, high_threshold)
            self.display_image(edges, self.label_equalize_output_3)

    def display_image(self, image, label):
        height, width = image.shape
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
