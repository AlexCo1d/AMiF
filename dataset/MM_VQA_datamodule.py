import lightning.pytorch as pl
from PIL import Image
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import transforms, InterpolationMode

from Utils.randaugment import RandomAugment
from dataset.VQA_Dataset import VQA_Dataset, VQA2019_Dataset


class VQADataModule(pl.LightningDataModule):
    def __init__(self, config, num_workers=18):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        image_size = config["image_size"]
        self.num_workers = num_workers
        dataset, data_path = config["dataset"], config["dataset_path"]
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([RandomAugment(2, 7, isPIL=True,
                                                  augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness',
                                                        'Sharpness',
                                                        'ShearX', 'ShearY', 'TranslateX',
                                                        'TranslateY', 'Rotate'])], p=0.8),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        # vqa_rad
        if dataset == 'radvqa':
            self.train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='VQA_RAD Image Folder',
                                             answer_list_flag=True,
                                             max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])
            self.test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='VQA_RAD Image Folder',
                                            answer_list_flag=True,
                                            max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])

        # pathvqa
        elif dataset == 'pathvqa':
            self.train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='images',
                                             answer_list_flag=True,
                                             max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])
            self.test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='images',
                                            answer_list_flag=True,
                                            max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])

        # slake
        elif dataset == 'slake':
            self.train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='imgs',
                                             answer_list_flag=True,
                                             max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])
            self.test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='imgs',
                                            answer_list_flag=True,
                                            max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])

        elif dataset == 'vqa2019':
            self.train_dataset = VQA2019_Dataset(data_path, train_transform, mode='train', img_root='images',
                                                 answer_list_flag=True,
                                                 max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])
            self.test_dataset = VQA2019_Dataset(data_path, test_transform, mode='test', img_root='images',
                                                answer_list_flag=True,
                                                max_prior_length=config['max_prior_length'], prior_mode=config['prior_mode'])

    def train_dataloader(self):
        c=self.test_dataset
        c.total_answers = self.train_dataset.total_answers
        return DataLoader(
            c,
            batch_size=self.config['batch_size'],
            num_workers=self.num_workers,
            shuffle=True,
            # drop_last=True,  # to keep shape consistent
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['valid']['val_batch_size'],
            num_workers=self.num_workers,
            shuffle=False,  # False for validation!!!
            pin_memory=True,
            # sampler=SequentialSampler(self.val_dataset)
        )
