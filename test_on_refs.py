import os

import torch
from PIL import Image
import pandas as pd
import tqdm

from MedCLIP.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from MedCLIP.medclip import MedCLIPProcessor


model_path = "ref75/med_clip.bin"
state_dict = torch.load(model_path)

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.load_state_dict(state_dict)
model.cuda()
model.eval()
processor = MedCLIPProcessor()

test_dataset = pd.read_csv('DATASET.csv')
test_dataset = test_dataset[test_dataset['split'] == 'test'].reset_index().drop(columns=['index'])
test_dataset = test_dataset[test_dataset['Image'].apply(lambda x:os.path.isfile(x))].reset_index().drop(columns=['index'])
test_dataset['Reference'] = [eval(ref) for ref in test_dataset['Reference']]

all_img_embs = torch.empty((0, 512), device=torch.device('cuda'))
for img in tqdm.tqdm(test_dataset['Image']):
    image = Image.open(img).convert('RGB')
    inputs = processor(
        text=None,
        images=[image],
        return_tensors='pt',
        padding=True
    )
    with torch.no_grad():
        img_embeds = model.encode_image(inputs['pixel_values'].cuda())
    all_img_embs = torch.cat((all_img_embs, img_embeds), dim=0)

all_refs = []
gt = []
for idx, ref_list in enumerate(test_dataset['Reference']):
    gt.extend([idx] * len(ref_list))
    all_refs.extend(ref_list)

all_ref_embs = torch.empty((0, 512), device=torch.device('cuda'))
for ref in tqdm.tqdm(all_refs):
    inputs = processor(
        text=[ref],
        return_tensors='pt',
        padding=True
    )
    with torch.no_grad():
        text_embeds = model.encode_text(inputs['input_ids'].cuda(), inputs['attention_mask'].cuda())

    all_ref_embs = torch.cat((all_ref_embs, text_embeds), dim=0)

at1 = at5 = at10 = 0

for idx, ref_emb in enumerate(all_ref_embs):
    logits = model.compute_logits(all_img_embs, text_emb=ref_emb)
    vals, preds = torch.topk(logits, 10)
    if preds[0] == gt[idx]: at1 += 1
    if gt[idx] in preds[:5]: at5 += 1
    if gt[idx] in preds: at10 += 1

print('I2T Recall@1 =', at1 / len(all_refs))
print('I2T Recall@5 =', at5 / len(all_refs))
print('I2T Recall@10 =', at10 / len(all_refs))

# at1 = at5 = at10 = 0
# for idx, row in enumerate(logits_per_text):
#     vals, preds = torch.topk(row, 10)
#     if preds[0] == idx: at1 += 1
#     if idx in preds[:5]: at5 += 1
#     if idx in preds: at10 += 1

# print('T2I Recall@1 =', at1 / len(test_dataset))
# print('T2I Recall@5 =', at5 / len(test_dataset))
# print('T2I Recall@10 =', at10 / len(test_dataset))