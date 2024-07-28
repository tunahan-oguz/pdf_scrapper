import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from train.dataset import MedClipDataset
from torch.utils.data import DataLoader
import tqdm

model_path = "only_captions/med_clip.bin"
state_dict = torch.load(model_path)

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(state_dict)
model.cuda()
model.eval()


dataset = MedClipDataset('DATASET.csv', 'test') + MedClipDataset('DATASET.csv', 'train')
loader = DataLoader(dataset, batch_size=18, collate_fn=MedClipDataset.collate_fn, drop_last=False)
at1 = at5 = at10 = 0

all_img_embs = torch.empty((0, 512), device=torch.device('cuda'))
all_text_embs = torch.empty((0, 512), device=torch.device('cuda'))
paths = []
for test_sample in tqdm.tqdm(loader):
    with torch.no_grad():
        img_embeds = model.encode_image(test_sample['pixel_values'].cuda())
        text_embeds = model.encode_text(test_sample['input_ids'].cuda(), test_sample['attention_mask'].cuda())
        all_img_embs = torch.cat((all_img_embs, img_embeds), dim=0)
        all_text_embs = torch.cat((all_text_embs, text_embeds), dim=0)
        paths.extend(test_sample['path'])

torch.save({
    'image_paths': paths,
    'image_embeddings': all_img_embs.cpu(),
    'text_embeddings': all_text_embs.cpu()
}, 'embeddings_with_paths.pth')