from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QWidget, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("471 DEMO")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text here")
        self.text_input.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.text_input.setFixedSize(200,200)
        self.text_btn = QPushButton('Find')
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)  
        self.left_arrow_button = QPushButton("<")
        self.right_arrow_button = QPushButton(">")

        self.left_arrow_button.clicked.connect(self.show_previous_image)
        self.right_arrow_button.clicked.connect(self.show_next_image)

        self.image_paths = ["download.jpeg"] 
        self.current_image_index = 0

        main_layout.addWidget(self.text_input)
        main_layout.addWidget(self.text_btn)
        main_layout.addWidget(self.left_arrow_button)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.right_arrow_button)

        # Load the first image
        self.load_image()

    def load_image(self):
        if self.image_paths:
            pixmap = QPixmap(self.image_paths[self.current_image_index])
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def show_previous_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.load_image()

    def show_next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.load_image()


if __name__ == '__main__':
    app = QApplication([])

    window = MainWindow()
    window.show()
    app.exec()
