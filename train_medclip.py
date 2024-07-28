from MedCLIP.medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from MedCLIP.medclip import MedCLIPProcessor
from MedCLIP.medclip.trainer import Trainer
from PIL import Image
from torch.utils.data import DataLoader
from train.dataset import MedClipDataset
from prettytable import PrettyTable

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
model.cuda()

table = PrettyTable()
table.field_names = ["Parameter Name", "Trainable"]

for name, param in model.named_parameters():
    trainable = "Yes" if param.requires_grad else "No"
    table.add_row([name, trainable])

print(table)

dataloader = DataLoader(MedClipDataset("DATASET.csv", 'train', ref_r=0.75),
                        batch_size=18, shuffle=True, pin_memory=True, collate_fn=MedClipDataset.collate_fn)

med_clip_trainer = Trainer()
med_clip_trainer.train(
    model=model,
    train_objectives= [(dataloader, model, 1)],
    epochs=120,
    output_path="ref75",
    save_steps=1000,
    save_best_model=False
)



