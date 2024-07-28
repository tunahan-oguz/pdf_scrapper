import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
import pandas as pd 
from MedCLIP.medclip import MedCLIPProcessor
from PIL import Image
import torchvision.transforms as transforms
import tqdm

# Load the model state dictionary
model_path = "med_clip.bin"
state_dict = torch.load(model_path)

# Initialize the model and load the state dictionary
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(state_dict)
model.cuda()
model.eval()

# Load the dataset
all_dataset = pd.read_csv('DATASET2.csv')

# Initialize the processor
processor = MedCLIPProcessor()

# Function to convert image path to tensor
def image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

images = []
texts = []
paths = []

for index, row in all_dataset.iterrows():
    image_path = row['Image']
    description = row['Cleaned_Description']
    image_tensor = image_to_tensor(image_path)
    if image_tensor is None:
        continue
    images.append(image_tensor)
    texts.append(description)
    paths.append(image_path)

processed_data = processor(
    text=texts, 
    images=images, 
    return_tensors="pt", 
    padding=True
)
all_img_embs = torch.empty((0, 512), device=torch.device('cuda'))
all_text_embs = torch.empty((0, 512), device=torch.device('cuda'))

for test_sample in tqdm.tqdm(processed_data):
    print('test ' , test_sample)
    with torch.no_grad():
        img_embeds = model.encode_image(test_sample['pixel_values'].cuda())
        text_embeds = model.encode_text(test_sample['input_ids'].cuda(), test_sample['attention_mask'].cuda())
        all_img_embs = torch.cat((all_img_embs, img_embeds), dim=0)
        all_text_embs = torch.cat((all_text_embs, text_embeds), dim=0)

