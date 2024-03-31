# Importing Packages
# importing module
import logging
import os
import sys
import timeit
from typing import Callable, Type

import cv2
import numpy as np
from PIL import Image
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, QFile, QTextStream
from PyQt5.QtWidgets import QMessageBox

from UI import mainGUI as m
from UI import breeze_resources



# Create and configure logger
logging.basicConfig(level=logging.DEBUG,
                    filename="app.log",
                    format='%(lineno)s - %(levelname)s - %(message)s',
                    filemode='w')

# Creating a logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ImageProcessor(m.Ui_MainWindow):
    """
    Main Class of the program GUI
    """

    def __init__(self, starter_window):
        """
        Main loop of the UI
        :param starter_window: QMainWindow Object
        """
        super(ImageProcessor, self).setupUi(starter_window)

        # Setup Main_TabWidget Connections
        self.Main_TabWidget.setCurrentIndex(0)
        self.tab_index = self.Main_TabWidget.currentIndex()
        self.Main_TabWidget.currentChanged.connect(self.tab_changed)

        

        

        


        # This contains all the widgets to setup them in one loop
        self.imageWidgets = [
                             self.img4_input, self.img4_output,
                             ]

        # Initial Variables
        self.currentNoiseImage = None
        self.edged_image = None
        self.filtered_image = None
        self.output_hist_image = None
        self.updated_image = None
        self.db_path = None

        # Threads and workers we will use in QThread
        # Used for SIFT Algorithm, Feature Matching and Median Filter
        self.threads = {}
        self.workers = {}

        # SIFT Results
        self.sift_results = {}

        # Dictionaries to store images data
        self.imagesData = {}
        self.imagesPaths = {}
        self.widths = {}
        self.heights = {}

        # Object of FaceRecognizer Class
        self.recognizer = None


        

        # Setup Load Buttons Connections
        self.btn_load_4_1.clicked.connect(lambda: self.load_file(self.tab_index))
        

      


        # Setup Active Contour Buttons
        self.btn_apply_contour.clicked.connect(self.active_contour)
        
     


        self.setup_images_view()

    def tab_changed(self):
        """
        Updates the current tab index

        :return: void
        """

        self.tab_index = self.Main_TabWidget.currentIndex()

    def setup_images_view(self):
        """
        This function is responsible for:
            - Adjusting the shape and scales of the widgets
            - Remove unnecessary options

        :return: void
        """

        for widget in self.imageWidgets:
            widget.ui.histogram.hide()
            widget.ui.roiBtn.hide()
            widget.ui.menuBtn.hide()
            widget.ui.roiPlot.hide()
            widget.getView().setAspectLocked(False)
            widget.view.setAspectLocked(False)

    def load_file(self, img_id: int, multi_widget: bool = False):
        """
        Load the File from User

        :param img_id: current tab index
        :param multi_widget: Flag to check if the tab has more than one image
        :return:
        """

        # Open File Browser
        logger.info("Browsing the files...")
        repo_path = "resources/Images"
        filename, file_format = QtWidgets.QFileDialog.getOpenFileName(None, "Load Image", repo_path,
                                                                      "*;;" "*.jpg;;" "*.jpeg;;" "*.png;;")

        # If the file is loaded successfully
        if filename != "":
            # Take last part of the filename string
            img_name = filename.split('/')[-1]

            # Read the image
            img_bgr = cv2.imread(filename)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Loading one image only in the tab
            if multi_widget is False:
                # Convert int index to key to use in the dictionaries
                img_idx = str(img_id) + "_1"
            # When Loading 2nd Image in same Tab (Hybrid, SIFT, Template Matching)
            else:
                img_idx = str(img_id) + "_2"

            # Store the image
            self.imagesData[img_idx] = img_rgb
            self.imagesPaths[img_idx] = filename
            self.heights[img_idx], self.widths[img_idx], _ = img_rgb.shape

         


            # Enable the comboBoxes and settings
            self.enable_gui(tab_id=img_id)

            

            logger.info(f"Added Image #{img_id}: {img_name} successfully")

       

    def enable_gui(self, tab_id: int):
        """
        This function enables the required elements in the gui
        :param tab_id: if of the current tab
        :return:
        """
        # in Active Contour Tab
        if tab_id == 4:
            self.contour_settings_layout.setEnabled(True)
            self.btn_apply_contour.setEnabled(True)
           



##cont

    def active_contour(self):
        """
    Apply Active Contour Model (Snake) to the given image on a certain shape.
    This algorithm is applied based on Greedy Algorithm
    :return:
    """

    # Check if the key '4_1' exists in the imagesData dictionary
        if '4_1' not in self.imagesData:
            logger.error("No image loaded for tab index 4")
            return

        # Get Contour Parameters
        alpha = float(self.text_alpha.text())
        beta = float(self.text_beta.text())
        gamma = float(self.text_gamma.text())
        num_iterations = int(self.text_num_iterations.text())
        num_points_circle = 65
        num_xpoints = 180
        num_ypoints = 180
        w_line = 1
        w_edge = 8

        # Initial variables
        contour_x, contour_y, window_coordinates = None, None, None

        # Calculate function run time
        start_time = timeit.default_timer()

        # Greedy Algorithm

        # copy the image because cv2 will edit the original source in the contour
        image_src = np.copy(self.imagesData["4_1"])

        # Create Initial Contour and display it on the GUI
        if self.radioButton_square_contour.isChecked():
            contour_x, contour_y, window_coordinates = Contour.create_square_contour(source=image_src,
                                                                                    num_xpoints=num_xpoints,
                                                                                    num_ypoints=num_ypoints)
            # Set parameters with pre-tested values for good performance
            alpha = 20
            beta = 0.01
            gamma = 2
            num_iterations = 60
            self.text_alpha.setText(str(alpha))
            self.text_beta.setText(str(beta))
            self.text_gamma.setText(str(gamma))
            self.text_num_iterations.setText(str(num_iterations))

        elif self.radioButton_circle_contour.isChecked():
            contour_x, contour_y, window_coordinates = Contour.create_elipse_contour(source=image_src,
                                                                                    num_points=num_points_circle)
            # Set parameters with pre-tested values for good performance
            alpha = 0.01
            beta = 0.01
            gamma = 2
            num_iterations = 50
            self.text_alpha.setText(str(alpha))
            self.text_beta.setText(str(beta))
            self.text_gamma.setText(str(gamma))
            self.text_num_iterations.setText(str(num_iterations))

        # Display the input image after creating the contour
        src_copy = np.copy(image_src)
        initial_image = self.draw_contour_on_image(src_copy, contour_x, contour_y)
        self.display_image(source=initial_image, widget=self.img4_input)

        # Calculate External Energy which will be used in each iteration of greedy algorithm
        external_energy = gamma * Contour.calculate_external_energy(image_src, w_line, w_edge)

        # Copy the coordinates to update them in the main loop
        cont_x, cont_y = np.copy(contour_x), np.copy(contour_y)

        # main loop of the greedy algorithm
        for iteration in range(num_iterations):
            # Start Applying Active Contour Algorithm
            cont_x, cont_y = Contour.iterate_contour(source=image_src, contour_x=cont_x, contour_y=cont_y,
                                                    external_energy=external_energy,
                                                    window_coordinates=window_coordinates,
                                                    alpha=alpha, beta=beta)

            # Display the new contour after each iteration
            src_copy = np.copy(image_src)
            processed_image = self.draw_contour_on_image(src_copy, cont_x, cont_y)
            self.display_image(source=processed_image, widget=self.img4_output)

            # Used to allow the GUI to update ImageView Object without lagging
            QtWidgets.QApplication.processEvents()

        # Function end
        end_time = timeit.default_timer()

        # Show only 5 digits after floating point
        elapsed_time = format(end_time - start_time, '.5f')
    

    @staticmethod
    def draw_contour_on_image(source, points_x, points_y):
        """
        This function draws a given contour coordinates on the given image

        :param source: image source to draw the contour above it
        :param points_x: list of indices of the contour in x-direction
        :param points_y: list of indices of the contour in y-direction
        :return:
        """

        # Copy the image source to prevent modifying the original image
        src = np.copy(source)

        points = []
        for px, py in zip(points_x, points_y):
            points.append([px, py])

        points = np.array(points, np.int32)
        points = points.reshape((-1, 1, 2))

        image = cv2.polylines(src, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        return image

    @staticmethod
    def display_image(source: np.ndarray, widget: pg.ImageView):
        """
        Displays the given image source in the specified ImageView widget

        :param source: image source
        :param widget: ImageView object
        :return: void
        """

        # Copy the original source because cv2 updates the passed parameter
        src = np.copy(source)

        # Rotate the image 90 degree because ImageView is rotated
        src = cv2.transpose(src)

        widget.setImage(src)
        widget.view.setRange(xRange=[0, src.shape[0]], yRange=[0, src.shape[1]],
                             padding=0)
        widget.ui.roiPlot.hide()


def main():
    """
    the application startup functions
    :return:
    """

    app = QtWidgets.QApplication(sys.argv)

    # set stylesheet
    file = QFile("UI/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    main_window = QtWidgets.QMainWindow()
    ImageProcessor(main_window)
    main_window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
