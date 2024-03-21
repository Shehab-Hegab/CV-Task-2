import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from mainwindow_ui import Ui_MainWindow  # Import the generated UI class

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
