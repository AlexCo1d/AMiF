import re
import time
from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image, ImageFilter
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import tokenizers
import random

from torchvision.transforms import InterpolationMode

from Utils.randaugment import RandomAugment


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MIMICDataset(Dataset):
    def __init__(
            self,
            data_root,
            transform,
            max_caption_length: int = 200,
            mv=False,
            preprocess=None
    ):
        self.mv = mv
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.transform = transform
        self.csv = self.read_csv()
        self.images_list, self.report_list = self.csv['image_path'], self.csv['report_content']
        if self.mv:
            self.view_type_list = self.csv['view_type']

        self.preprocess = preprocess
        # self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        # self.tokenizer.enable_padding(length=self.max_caption_length)

    def __len__(self):
        return len(self.images_list)

    def _random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1] - 1):
            if masked_tokens[0][i] == 0:
                break

            if masked_tokens[0][i - 1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue

            if masked_tokens[0][i - 1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        if self.mv:
            # random select multivew image from same study:
            sent = self.report_list[index]
            text = pre_caption(sent, self.max_caption_length)

            image_list = self.images_list[index].split(';')
            if len(image_list) > 1:
                view_type_list = self.view_type_list[index].split(';')
                index1, index2 = select_two_index(view_type_list)
                image1_ = pil_loader(image_list[index1])
                image2_ = pil_loader(image_list[index2])
                image1 = self.transform(image1_)
                image2 = self.transform(image2_)
                d = {
                    "_image1": self.preprocess(image1_),
                    "_image2": self.preprocess(image2_),
                    "image1": image1,
                    "image2": image2,
                    # 'false_image': false_image,
                    "text": text
                }
                return d
            else:
                image_ = pil_loader(image_list[0])
                image1 = self.transform(image_)
                image2 = self.transform(image_)
                t = self.preprocess(image_)
                d = {
                    "_image1": t,
                    "_image2": t,
                    "image1": image1,
                    "image2": image2,
                    # 'false_image': false_image,
                    "text": text
                }
                return d
            # false_image = get_false_image(self.images_list, self.transform)

        else:
            image = pil_loader(self.images_list[index])
            image = self.transform(image)
            sent = self.report_list[index]
            text = pre_caption(sent, self.max_caption_length)
            return {
                "image1": image,
                "text": text
            }

    def read_csv(self):
        if self.mv:
            csv_path = os.path.join(self.data_root, 'training_mv.csv')
            df = pd.read_csv(csv_path, sep=',')
            return df
        else:
            csv_path = os.path.join(self.data_root, 'training.csv')
            df = pd.read_csv(csv_path, sep=',')
            return df

    # def collate_fn(self, instances: List[Tuple]):
    #     image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], []
    #     # flattern
    #     for b in instances:
    #         image, ids, attention_mask, type_ids, masked_ids = b
    #         image_list.append(image)
    #         ids_list.append(ids)
    #         attention_mask_list.append(attention_mask)
    #         type_ids_list.append(type_ids)
    #         masked_ids_list.append(masked_ids)
    #
    #     # stack
    #     image_stack = torch.stack(image_list)
    #     ids_stack = torch.stack(ids_list).squeeze()
    #     attention_mask_stack = torch.stack(attention_mask_list).squeeze()
    #     type_ids_stack = torch.stack(type_ids_list).squeeze()
    #     masked_ids_stack = torch.stack(masked_ids_list).squeeze()
    #
    #     # sort and add to dictionary
    #     return_dict = {
    #         "image_1": image_stack,
    #         "labels": ids_stack,
    #         "attention_mask": attention_mask_stack,
    #         "type_ids": type_ids_stack,
    #         "ids": masked_ids_stack
    #     }
    #
    #     return return_dict


def pre_caption(caption, max_words):
    caption = re.sub(
        r"([_,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption


def select_two_index(view_type_list):
    # 将类型和索引映射到字典中
    type_dict = {}
    for idx, view_type in enumerate(view_type_list):
        if view_type in type_dict:
            type_dict[view_type].append(idx)
        else:
            type_dict[view_type] = [idx]

    # 获取所有类型的列表
    types = list(type_dict.keys())

    # 如果只有一个类型可用
    if len(types) == 1:
        return random.sample(range(len(view_type_list)), 2)

    # 尝试随机选取两个不同的类型
    chosen_types = random.sample(types, 2)
    index1 = random.choice(type_dict[chosen_types[0]])
    index2 = random.choice(type_dict[chosen_types[1]])

    # 确保不是从相同的类型中选择的两个相同的索引
    while index1 == index2:
        index2 = random.choice(type_dict[chosen_types[1]])

    return index1, index2


class MediCaTDataset(Dataset):
    def __init__(
            self,
            data_root,
            transform,
            img_root='figures',
            max_caption_length: int = 200,
            mv=False,
            preprocess=None
    ):
        self.mv = mv
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.img_root = img_root

        self.transform = transform

        self.data = self.read_json()
        self.images_list = self.data['pdf_hash'] + '_' + self.data['fig_uri']
        self.texts_list = self.data['s2_caption']
        self.preprocess = preprocess
        # self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        # self.tokenizer.enable_padding(length=self.max_caption_length)

    def read_json(self):
        json_path = os.path.join(self.data_root, 'training.json')
        df = pd.read_json(json_path, lines=True)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.images_list[index]
        image = pil_loader(os.path.join(self.data_root, self.img_root, image))
        sent = self.texts_list[index]
        text = pre_caption(sent, self.max_caption_length)
        image1 = self.transform(image)
        t = self.preprocess(image)
        # false_image = get_false_image(self.images_list, self.transform, root=os.path.join(self.data_root, self.img_root))
        if self.mv:
            image2 = self.transform(image)
            # random select multivew image from same study:
            return {
                "_image1": t,
                "_image2": t,
                "image1": image1,
                "image2": image2,
                "text": text,
                # "false_image": false_image
            }
        else:
            return {
                "image1": image,
                "text": text
            }


class ROCODataset(Dataset):
    def __init__(
            self,
            data_root,
            transform,
            img_root='images',
            max_caption_length: int = 200,
            mv=False,
            split='train',
            preprocess=None,
    ):
        self.mv = mv
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.img_root = img_root
        self.transform = transform

        self.data = self.read_csv(split=split)
        self.images_list = self.data['image_path']
        self.texts_list = self.data['text']
        # self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        # self.tokenizer.enable_padding(length=self.max_caption_length)
        self.preprocess = preprocess

    def read_csv(self, split='train'):
        if split == 'train':
            csv_path = os.path.join(self.data_root, 'training.csv')
            df = pd.read_csv(csv_path, sep=',')
        else:
            csv_path = os.path.join(self.data_root, 'test.csv')
            df = pd.read_csv(csv_path, sep=',').sample(frac=1, random_state=42).reset_index(drop=True)[:2000]
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.images_list[index]
        image = pil_loader(os.path.join(self.data_root, self.img_root, image))
        sent = self.texts_list[index]
        text = pre_caption(sent, self.max_caption_length)
        image1 = self.transform(image)
        t = self.preprocess(image)
        # false_image = get_false_image(self.images_list, self.transform, root=os.path.join(self.data_root, self.img_root))
        if self.mv:
            image2 = self.transform(image)
            # random select multivew image from same study:
            return {
                "_image1": t,
                "_image2": t,
                "image1": image1,
                "image2": image2,
                "text": text,
                # "false_image": false_image
            }
        else:
            return {
                "image1": image,
                "text": text
            }


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_false_image(images_list, transform, root=''):
    random_index = random.randint(0, len(images_list) - 1)
    l = images_list[random_index].split(';')
    r = random.randint(0, len(l) - 1)
    false_image = pil_loader(os.path.join(root, l[r]))
    false_image = transform(false_image)
    return false_image


def get_pretrain_dataset(args):
    img_size = args.img_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.6),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([RandomAugment(2, 6, isPIL=True, augs=['Identity', 'Equalize', 'Sharpness',
                                                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
                                                                      'Rotate'])], p=0.8),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    MIMIC = MIMICDataset(data_root=args.data_path, transform=transform, mv=args.mv)
    fpath = os.path.dirname(args.data_path)
    if args.concat:
        # capture the upper path of args.datapath
        MediCaT = MediCaTDataset(data_root=os.path.join(fpath, 'medicat'), transform=transform, mv=args.mv)
        ROCO = ROCODataset(data_root=os.path.join(fpath, 'roco-dataset'), transform=transform, mv=args.mv)
        return ConcatDataset([MIMIC, MediCaT, ROCO])
    else:
        return ROCODataset(data_root=os.path.join(fpath, 'roco-dataset'), transform=transform, mv=args.mv, split='test')
