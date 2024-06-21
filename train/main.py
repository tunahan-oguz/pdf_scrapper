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
from train.model.VSE import VSE
from train.vocab import Vocabulary
import tqdm
import numpy as np
import torch


__arch__ = {
    "VSE": VSE,
}
__rundir__ = "./run"

min_loss = np.float32('inf')

def train_one(model, loader, criterion, optim, scheduler, logger):
    model = model.cuda()
    model.train()
    accumulated_loss = 0
    step = 0
    for im, text, lens in tqdm.tqdm(loader):
        step += 1
        optim.zero_grad()
 
        im = im.cuda(); text = text.cuda()
        outs = model(im, text, lens)
 
        loss = criterion(*outs) # contrastive
        loss.backward()
        optim.step()
        accumulated_loss += loss.item()
 
        if step % 15 == 0:
            logger.log({
                f"loss": accumulated_loss / 50
            })
            accumulated_loss = 0
        logger.log({
            f"learning_rate" : scheduler.get_last_lr()[0]
        })
    scheduler.step()

    
def validate(model, loader, criterion):

    model.eval()
    total_loss = 0
    for im, text, lens in tqdm.tqdm(loader):
        im = im.cuda(); text = text.cuda()
        with torch.no_grad():
            outs = model(im, text, lens)
            loss = criterion(*outs) # contrastive
            total_loss += loss.item()
    model_selection(model, total_loss)


def model_selection(model : torch.nn.Module, total_loss):
    global min_loss
    if total_loss < min_loss:
        torch.save(model.state_dict(), os.path.join(__rundir__, wandb.run.name, "best.pth"))
        min_loss = total_loss
        print(f"Best model saved to {os.path.join(__rundir__, wandb.run.name)}")

def loop(epochs, model, train_loader, criterion, optim, scheduler, logger, val_loader):
    for e in range(epochs):
        print(f"[TRAIN: {e}]")
        train_one(model, train_loader, criterion, optim, scheduler, logger)
        print(f"[VAL: {e}]")
        validate(model, val_loader, criterion)


@hydra.main(version_base=None, config_path="configs", config_name="vse")
def main(config: DictConfig):
    global __rundir__
    # Load vocabulary
    vocab_path = to_absolute_path(config.paths.vocab_path)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    # Update model parameters with vocab size
    model_params = {
        "embed_size": config.model.embed_size,
        "finetune": config.model.finetune,
        "cnn_type": config.model.cnn_type,
        "use_abs": config.model.use_abs,
        "no_imgnorm": config.model.no_imgnorm,
        "vocab_size": len(vocab),
        "word_dim": config.model.word_dim,
        "num_layers": config.model.num_layers
    }

    EPOCH = config.training.epoch
    # Initialize model, optimizer, and criterion
    model = __arch__[config.arch](**model_params)
    optim = AdamW(model.parameters(), lr=config.training.lr)
    scheduler = CosineAnnealingLR(optim, T_max=EPOCH // 5)
    criterion = ContrastiveLoss()

    # Create datasets and dataloaders
    dataset_root = to_absolute_path(config.paths.dataset_root)
    train_dataset = AnatomDataset(root=dataset_root, split="train", vocab=vocab, transform=IMAGE_TRANSFORMS)
    valid_dataset = AnatomDataset(root=dataset_root, split="valid", vocab=vocab, transform=IMAGE_TRANSFORMS)
    test_dataset = AnatomDataset(root=dataset_root, split="test", vocab=vocab, transform=IMAGE_TRANSFORMS)

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
    
    loop(EPOCH, model, train_dataloader, criterion, optim, scheduler, logger, valid_dataloader)

    wandb.run.finish()

if __name__ == "__main__":
    main()
