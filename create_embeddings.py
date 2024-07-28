import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
import pandas as pd 
from MedCLIP.medclip import MedCLIPProcessor
from PIL import Image
import torchvision.transforms as transforms

# Load the model state dictionary
model_path = "med_clip.bin"
state_dict = torch.load(model_path)

# Initialize the model and load the state dictionary
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

# Load the dataset
all_dataset = pd.read_csv('DATASET.csv')

# Initialize the processor
processor = MedCLIPProcessor()

# Function to convert image path to tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    return image_tensor

images = []
texts = []

for index, row in all_dataset.iterrows():
    image_path = row['Image']
    description = row['Cleaned_Description']
    
    image_tensor = image_to_tensor(image_path)
    images.append(image_tensor)
    texts.append(description)

processed_batch = processor(
    text=texts, 
    images=images, 
    return_tensors="pt", 
    padding=True
)



print(processed_batch)

