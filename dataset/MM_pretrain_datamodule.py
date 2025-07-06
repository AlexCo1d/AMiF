import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, ConcatDataset, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor

from Retrieval.retrieval_dataset import retrieval_dataset
from Utils import misc
from Utils.randaugment import RandomAugment
from dataset.pretrain_datasets import GaussianBlur, MIMICDataset, MediCaTDataset, ROCODataset


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config, num_workers=20):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.save_hyperparameters()
        self.config = config
        self.num_workers = num_workers
        self.data_path = self.config['data_path']
        self.image_size = config["image_size"]
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.6),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([RandomAugment(2, 4, isPIL=True, augs=['Identity', 'Equalize', 'Sharpness',
                                                                          'ShearX', 'ShearY', 'TranslateX',
                                                                          'TranslateY',
                                                                          'Rotate'])], p=0.8),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
        if self.config['softlabel'] is True:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.preprocess = None

    def setup(self, stage='train') -> None:
        if stage == 'train':
            MIMIC = MIMICDataset(data_root=self.data_path, transform=self.train_transform,
                                 mv=self.config['multiview'], preprocess=self.preprocess)
            self.train_dataset = MIMIC
            fpath = os.path.dirname(self.data_path)
            if self.config['concat'] is True:
                MediCaT = MediCaTDataset(data_root=os.path.join(fpath, 'medicat'), transform=self.train_transform,
                                         mv=self.config['multiview'], preprocess=self.preprocess)
                ROCO = ROCODataset(data_root=os.path.join(fpath, 'roco-dataset'), transform=self.train_transform,
                                   mv=self.config['multiview'], preprocess=self.preprocess)
                self.train_dataset = ConcatDataset([MIMIC, MediCaT, ROCO])

        self.val_dataset = retrieval_dataset(os.path.join(fpath, self.config['valid']['dataset']),
                                             self.val_transform)

    def train_dataloader(self):
        # num_tasks = misc.get_world_size()
        # global_rank = misc.get_rank()
        # sampler_train = torch.utils.data.DistributedSampler(
        #     self.train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,  # to keep shape consistent
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['valid']['val_batch_size'],
            num_workers=self.num_workers,
            shuffle=False,  # False for validation!!!
            pin_memory=True,
            # sampler=SequentialSampler(self.val_dataset)
        )