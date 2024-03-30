import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from mainwindow_ui import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_equalize_1.clicked.connect(self.load_image)
        self.ui.comboBox.currentIndexChanged.connect(self.process_image)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.display_image(pixmap, self.ui.label_equalize_input_3)
            self.current_image_path = file_path
            self.process_image()

    def display_image(self, image, label):
        label.setPixmap(image.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setScaledContents(True)

    def process_image(self):
        if hasattr(self, 'current_image_path'):
            method = self.ui.comboBox.currentText()
            if method == "Canny":
                input_image = self.ui.label_equalize_input_3.pixmap().toImage().convertToFormat(QImage.Format_Grayscale8)
                image_data = input_image.bits().asarray(input_image.width() * input_image.height())
                input_image_array = np.frombuffer(image_data, dtype=np.uint8).reshape((input_image.height(), input_image.width()))
                edges = self.canny_edge_detector(input_image_array)
                q_image = QImage(edges.data, edges.shape[1], edges.shape[0], edges.shape[1], QImage.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
                self.display_image(pixmap, self.ui.label_equalize_output_3)
            else:
                # Do nothing or apply other methods
                pass

    def gaussian_blur(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def sobel_filter(self, image):
        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])

        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

        gradient_x = cv2.filter2D(image, -1, kernel_x)
        gradient_y = cv2.filter2D(image, -1, kernel_y)

        magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        magnitude *= 255.0 / magnitude.max()
        orientation = np.arctan2(gradient_y, gradient_x)

        return magnitude, orientation

    def non_max_suppression(self, magnitude, orientation):
        suppressed = np.zeros_like(magnitude)
        angle = orientation * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, magnitude.shape[0] - 1):
            for j in range(1, magnitude.shape[1] - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def hysteresis_thresholding(self, image, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
        high_threshold = image.max() * high_threshold_ratio
        low_threshold = high_threshold * low_threshold_ratio

        strong_edges = (image > high_threshold)
        weak_edges = (image >= low_threshold) & (image <= high_threshold)

        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if weak_edges[i, j]:
                    if strong_edges[i - 1:i + 2, j - 1:j + 2].any():
                        strong_edges[i, j] = True
                    else:
                        weak_edges[i, j] = False

        edges = np.zeros_like(image)
        edges[strong_edges] = 255

        return edges

    def canny_edge_detector(self, image, kernel_size=5, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
        blurred_image = self.gaussian_blur(image, kernel_size)

        magnitude, orientation = self.sobel_filter(blurred_image)
        suppressed = self.non_max_suppression(magnitude, orientation)
        edges = self.hysteresis_thresholding(suppressed, low_threshold_ratio, high_threshold_ratio)
        return edges

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
