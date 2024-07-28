import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import nltk
import pickle
from train.embed.vocab import Vocabulary
from train.image_transform import IMAGE_TRANSFORMS
from MedCLIP.medclip import MedCLIPProcessor
from nltk.tokenize import word_tokenize

class AnatomDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """

    def __init__(self, root, split, vocab, transform=IMAGE_TRANSFORMS,
                 desc_set = "Description", ref_r = 0.0):
        self.root = root
        self.vocab = vocab
        self.split = split
        self.transform = transform
        self.dataset = pd.concat([pd.read_csv(os.path.join(root, csv_file)) for csv_file in os.listdir(root)], ignore_index=True)
        self.ids = []
        self.dataset = self.dataset[self.dataset['split'] == split].reset_index().drop(columns=['index'])
        self.desc_set = desc_set
        self.ref_r = ref_r

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        vocab = self.vocab

        image = Image.open(self.dataset['Image'][index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        r = np.random.random()
        caption = self.dataset[self.desc_set][index]
        if r < self.ref_r:
            # give the reference sentence as the caption
            refs = eval(self.dataset['Reference'][index])
            caption = refs[np.random.randint(0, len(refs))] if refs else caption
            
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.tensor(caption, dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def collate_fn(data):
        """Build mini-batch tensors from a list of (image, caption) tuples.
        Args:
            data: list of (image, caption) tuple.
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        # Sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, captions = zip(*data)

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths
    
# with open("vocab/simple_vocab.pkl", 'rb') as f:
#     vocab = pickle.load(f)
# dataset = AnatomDataset(root="dataset/descriptions", split="train", vocab=vocab, transform=IMAGE_TRANSFORMS)
# dl = data.DataLoader(dataset=dataset, batch_size=4, collate_fn=AnatomDataset.collate_fn)
# print(dl)

class MedClipDataset(data.Dataset):
    """
    Dataset loader for Flickr30k and Flickr8k full datasets.
    """
    train = None
    processor = MedCLIPProcessor()
    def __init__(self, csv, split, transform=IMAGE_TRANSFORMS,
                 desc_set = "Description", ref_r = 0.0):
        self.split = split
        use_val = 'valid' if split == 'train' else False
        self.transform = transform
        self.dataset = pd.read_csv(csv)
        self.ids = []
        if use_val:
            self.dataset = self.dataset[(self.dataset['split'] == split) | (self.dataset['split'] == use_val)].reset_index().drop(columns=['index'])
        else:
            self.dataset = self.dataset[self.dataset['split'] == split].reset_index().drop(columns=['index'])
        # because of cleaning
        self.dataset = self.dataset[self.dataset['Image'].apply(lambda x:os.path.isfile(x))].reset_index().drop(columns=['index'])
        self.desc_set = desc_set
        self.ref_r = ref_r
        MedClipDataset.train = split == 'train'

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """

        image = Image.open(self.dataset['Image'][index]).convert('RGB')
        r = np.random.random()
        caption = self.dataset[self.desc_set][index]
        if r < self.ref_r:
            # give the reference sentence as the caption
            refs = eval(self.dataset['Reference'][index]) if isinstance(self.dataset['Reference'][index], str) else self.dataset['Reference'][index]
            caption = refs[np.random.randint(0, len(refs))] if refs else caption
            
        return {"img": image, "caption": caption,"path":self.dataset['Image'][index]}

    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def collate_fn(data):
        """Build mini-batch tensors from a list of (image, caption) tuples.
        Args:
            data: list of (image, caption) tuple.
                - image: torch tensor of shape (3, 256, 256).
                - caption: torch tensor of shape (?); variable length.

        Returns:
            images: torch tensor of shape (batch_size, 3, 256, 256).
            targets: torch tensor of shape (batch_size, padded_length).
            lengths: list; valid length for each padded caption.
        """
        images = [d['img'] for d in data]
        texts = [d['caption'] for d in data]
        paths = [d['path'] for d in data]
        processed_batch =  MedClipDataset.processor(
            text=texts, 
            images=images, 
            return_tensors="pt", 
            padding=True
        )
        processed_batch['return_loss'] = MedClipDataset.train
        processed_batch['original_text'] = texts
        processed_batch['path'] = paths
        return processed_batch
    
