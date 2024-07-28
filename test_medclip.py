import torch
from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from train.dataset import MedClipDataset
from torch.utils.data import DataLoader
import tqdm

model_path = "ref75/med_clip.bin"
state_dict = torch.load(model_path)

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(state_dict)
model.cuda()
model.eval()


test_dataset = MedClipDataset('DATASET.csv', 'test')
loader = DataLoader(test_dataset, batch_size=18, collate_fn=MedClipDataset.collate_fn)
at1 = at5 = at10 = 0

all_img_embs = torch.empty((0, 512), device=torch.device('cuda'))
all_text_embs = torch.empty((0, 512), device=torch.device('cuda'))
for test_sample in tqdm.tqdm(loader):
    with torch.no_grad():
        img_embeds = model.encode_image(test_sample['pixel_values'].cuda())
        text_embeds = model.encode_text(test_sample['input_ids'].cuda(), test_sample['attention_mask'].cuda())
        
        all_img_embs = torch.cat((all_img_embs, img_embeds), dim=0)
        all_text_embs = torch.cat((all_text_embs, text_embeds), dim=0)


logits_per_image = model.compute_logits(all_img_embs, all_text_embs)
logits_per_text = logits_per_image.t()

for idx, row in enumerate(logits_per_image):
    vals, preds = torch.topk(row, 10)
    if preds[0] == idx: at1 += 1
    if idx in preds[:5]: at5 += 1
    if idx in preds: at10 += 1


print('I2T Recall@1 =', at1 / len(test_dataset))
print('I2T Recall@5 =', at5 / len(test_dataset))
print('I2T Recall@10 =', at10 / len(test_dataset))

at1 = at5 = at10 = 0
for idx, row in enumerate(logits_per_text):
    vals, preds = torch.topk(row, 10)
    if preds[0] == idx: at1 += 1
    if idx in preds[:5]: at5 += 1
    if idx in preds: at10 += 1

print('T2I Recall@1 =', at1 / len(test_dataset))
print('T2I Recall@5 =', at5 / len(test_dataset))
print('T2I Recall@10 =', at10 / len(test_dataset))