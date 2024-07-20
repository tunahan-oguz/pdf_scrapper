import pickle
import os
import torch.utils.data as data
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import to_absolute_path
from train.dataset import AnatomDataset
from train.image_transform import IMAGE_TRANSFORMS
from train.loss import ContrastiveLoss
import train.model as models
from train.embed.vocab import Vocabulary
import train.eval as eval
import tqdm
import numpy as np
import torch


__arch__ = {
    "VSE": models.VSE.VSE,
}
__rundir__ = "./run"



cfg = None

max_r1 = np.float32('inf') * -1

def train_one(model, loader, criterion, optim, scheduler, logger):
    global cfg

    model = model.cuda()
    model.train()
    accumulated_loss = 0
    step = 0
    for im, text, lens in tqdm.tqdm(loader):
        step += 1
        optim.zero_grad()
 
        im = im.cuda(); text = text.cuda()
        img_emb, cap_emb = model(im, text, lens)
 
        loss = criterion(img_emb, cap_emb) # contrastive
        loss.backward()
        optim.step()
        accumulated_loss += loss.item()
 
        if step % cfg.log_step == 0:
            logger.log({
                "train":
                {
                    f"loss": accumulated_loss / cfg.log_step,
                    f"learning_rate" : scheduler.get_last_lr()[0]
                }
            })
            accumulated_loss = 0
    scheduler.step()

    
def validate(model, loader, logger, measure='cosine'):
    model.eval()
    
    img_embs, cap_embs = eval.encode_data(model, loader, log_step=10)
    (r1, r5, r10,) = eval.i2t(img_embs, cap_embs, measure=measure)
    (r1i, r5i, r10i) = eval.t2i(img_embs, cap_embs, measure=measure)

    logger.log({
        "validation":
        {
            f"i2t-Recall@1": r1,
            f"i2t-Recall@5": r5,
            f"i2t-Recall@10": r10,
        }
    })
    logger.log({
        "validation":
        { 
            f"t2i-Recall@1": r1i,
            f"t2i-Recall@5": r5i,
            f"t2i-Recall@10": r10i,
        }    
})

    model_selection(model, r1)


def model_selection(model : torch.nn.Module, r1):
    global max_r1
    if max_r1 < r1:
        torch.save(model.state_dict(), os.path.join(__rundir__, wandb.run.name, "best.pth"))
        max_r1 = r1
        print(f"Best model saved to {os.path.join(__rundir__, wandb.run.name)}")

def test(model : torch.nn.Module, test_loader, logger):
    global cfg
    global __rundir__
    weights = torch.load(os.path.join(__rundir__, wandb.run.name, "best.pth"))
    model.eval()
    model.load_state_dict(weights)
    
    img_embs, cap_embs = eval.encode_data(model, test_loader, log_step=10)
    (r1, r5, r10) = eval.i2t(img_embs, cap_embs, measure=cfg.sim_measure)
    (r1i, r5i, r10i) = eval.t2i(img_embs, cap_embs, measure=cfg.sim_measure)

    logger.log({
        "test":
        {
            f"i2t-Recall@1": r1,
            f"i2t-Recall@5": r5,
            f"i2t-Recall@10": r10,
        }
    })
    logger.log({
        "test":
        { 
            f"t2i-Recall@1": r1i,
            f"t2i-Recall@5": r5i,
            f"t2i-Recall@10": r10i,
        }    
})

def loop(epochs, model, train_loader, criterion, optim, scheduler, logger, val_loader, test_loader):
    global cfg
    for e in range(epochs):
        print(f"[TRAIN: {e}]")
        train_one(model, train_loader, criterion, optim, scheduler, logger)
        print(f"[VAL: {e}]")
        validate(model, val_loader, logger, measure=cfg.sim_measure)
    test(model, test_loader, logger)

@hydra.main(version_base=None, config_path="configs", config_name="vse")
def main(config: DictConfig):
    global __rundir__
    global cfg
    cfg = config
    # Load vocabulary
    vocab_path = to_absolute_path(config.paths.vocab_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Update model parameters with vocab size
    model_params = dict(config.model)
    model_params["vocab_size"] = len(vocab)

    EPOCH = config.training.epoch
    # Initialize model, optimizer, and criterion
    model = __arch__[config.arch](**model_params)
    optim = AdamW(model.parameters(), lr=config.training.lr)
    scheduler = CosineAnnealingLR(optim, T_max=EPOCH // 5)
    criterion = ContrastiveLoss()

    # Create datasets and dataloaders
    dataset_root = to_absolute_path(config.paths.dataset_root)
    train_dataset = AnatomDataset(root=dataset_root, split="train", vocab=vocab, transform=IMAGE_TRANSFORMS, desc_set=cfg.desc_set, ref_r=config.ref_inclusion.train)
    valid_dataset = AnatomDataset(root=dataset_root, split="valid", vocab=vocab, transform=IMAGE_TRANSFORMS, desc_set=cfg.desc_set, ref_r=config.ref_inclusion.val)
    test_dataset = AnatomDataset(root=dataset_root, split="test", vocab=vocab, transform=IMAGE_TRANSFORMS, desc_set=cfg.desc_set, ref_r=config.ref_inclusion.test)

    train_dataloader = data.DataLoader(
        dataset=train_dataset, 
        batch_size=config.training.batch_size, 
        collate_fn=AnatomDataset.collate_fn,
        num_workers=config.training.num_workers, 
        pin_memory=True, 
        drop_last=False, 
        shuffle=True
    )

    valid_dataloader = data.DataLoader(
        dataset=valid_dataset, 
        batch_size=config.training.batch_size, 
        collate_fn=AnatomDataset.collate_fn,
        num_workers=config.training.num_workers, 
        pin_memory=True, 
        drop_last=False, 
        shuffle=False
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset, 
        batch_size=config.training.batch_size, 
        collate_fn=AnatomDataset.collate_fn,
        num_workers=config.training.num_workers, 
        pin_memory=True, 
        drop_last=False, 
        shuffle=False
    )

    # Initialize WandB logger
    logger = wandb.init(
        project=config.wandb.project,
        save_code=True
    )
    os.makedirs(os.path.join(__rundir__, wandb.run.name), exist_ok=True)
    OmegaConf.save(config, os.path.join(__rundir__, wandb.run.name, "config.yaml"))
    
    loop(EPOCH, model, train_dataloader, criterion, optim, scheduler, logger, valid_dataloader, test_dataloader)

    wandb.run.finish()

if __name__ == "__main__":
    main()
