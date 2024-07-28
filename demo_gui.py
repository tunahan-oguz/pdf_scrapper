from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel, QWidget, QPushButton
import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from MedCLIP.medclip import MedCLIPProcessor


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
        self.text_input.setFixedSize(200, 200)
        self.text_btn = QPushButton('Find')
        self.image_label = QLabel()
        self.image_label.setFixedSize(400, 400)
        self.left_arrow_button = QPushButton("<")
        self.right_arrow_button = QPushButton(">")

        self.left_arrow_button.clicked.connect(self.show_previous_image)
        self.right_arrow_button.clicked.connect(self.show_next_image)
        self.text_btn.clicked.connect(self.new_query)
        self.image_paths = ["download.jpeg"]
        self.current_image_index = 0

        main_layout.addWidget(self.text_input)
        main_layout.addWidget(self.text_btn)
        main_layout.addWidget(self.left_arrow_button)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.right_arrow_button)

        # Load the first image
        self.load_image()

        # Load the model
        self.load_model()
        self.load_embeddings()
        

    def load_embeddings(self):
        embeddings_path = 'embeddings_with_paths.pth'
        data = torch.load(embeddings_path)
        self.saved_image_paths = data['image_paths']
        self.image_embeddings = data['image_embeddings'].cuda()
        print(len(self.image_embeddings))
        self.text_embeddings = data['text_embeddings'].cuda()

    def load_model(self):
        model_path = "med_clip.bin"
        state_dict = torch.load(model_path)

        # Initialize the model and load the state dictionary
        self.model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()

        # Initialize the processor
        self.processor = MedCLIPProcessor()

    def generate_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()

        with torch.no_grad():
            text_embeds = self.model.encode_text(input_ids, attention_mask)

        return text_embeds
    

    def find_similar_images(self,text_emb):
        similarities = self.model.compute_logits(img_emb=self.image_embeddings, text_emb=text_emb)
        similarities = similarities.flatten()
        try:
            top_k_indices = similarities.topk(5).indices.tolist()
        except Exception as e :
            print(e)
            return

        return [self.saved_image_paths[idx] for idx in top_k_indices]

    def new_query(self):
        query = self.text_input.toPlainText()
        text_embedding = self.generate_text_embedding(query)
        similar_image_paths = self.find_similar_images(text_embedding)
        self.image_paths = similar_image_paths
        self.current_image_index = 0
        self.load_image()

    def load_image(self):
        if self.image_paths:

            pixmap = QPixmap(self.image_paths[self.current_image_index][0])
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
