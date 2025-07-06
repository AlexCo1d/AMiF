import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import json
import PIL
from transformers import BertTokenizer, LlamaTokenizer

from Utils.randaugment import RandomAugment
from PIL import Image


class VQA_Dataset(Dataset):
    def __init__(self, data_path, transform, img_root='',
                 max_txt_length=512, mode='train', answer_list_flag: bool = False):
        if os.path.exists(os.path.join(data_path, f'{mode}.json')):
            with open(os.path.join(data_path, f'{mode}.json')) as f:
                self.data = json.load(f)
        self.mode = mode
        self.transform = transform
        self.data_path = data_path
        self.img_root = img_root
        self.max_txt_length = max_txt_length
        self.classifier_vqa = answer_list_flag
        answer_list = [pre_answer(self.data[i]['answer']) for i in range(len(self.data))]
        if self.mode == 'test':
            self.answer_list = list(set(answer_list))
            # self.answer_label = {answer: i for i, answer in enumerate(self.answer_list)}



        # answer_list = [item['answer'] for item in self.data]
        # make it unique.
        # self.answer_list = list(dict.fromkeys(answer_list))
        # self.tokenizer.enable_padding(length=max_answer_length)
        # self.tokenizer.enable_truncation(max_length=max_answer_length)
        # self.answer_list_ids = torch.stack(
        #     [torch.tensor(self.tokenizer.encode('[CLS] ' + item + ' sep').ids) for item in answer_list])
        # self.answer_list_att = torch.stack(
        #     [torch.tensor(self.tokenizer.encode('[CLS] ' + item + ' sep').attention_mask) for item in answer_list])
        # self.tokenizer.enable_truncation(max_length=max_caption_length)
        # self.tokenizer.enable_padding(length=max_caption_length)

    def __len__(self):
        return len(self.data)

    def random_answer(self, Question, Answer):
        Answer = str(Answer)
        pre_text = 'Question: ' + Question + ' The Answer is:'
        final_o = 'Question: ' + Question + ' The Answer is: ' + Answer
        return pre_text, final_o

    def __getitem__(self, idx):
        sample = self.data[idx]
        Question = sample['question']
        Answer = sample['answer']
        Answer = pre_answer(Answer)
        at = 'CLOSED' if (Answer == 'yes' or Answer == 'no') else 'OPEN'
        Question = pre_question(Question)
        ##### read image pathes #####
        img_path = os.path.join(self.data_path, self.img_root, sample['image_name'])
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        if self.classifier_vqa:
            pre_text = Question
        else:
            pre_text, final_o = self.random_answer(Question, Answer)
        # final_o = self.tokenizer(pre_text, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # input_ids = final_o.input_ids
        # attention_mask = final_o.attention_mask
        # input_ids = torch.tensor(input_ids).unsqueeze(0)
        # attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        # label = self.tokenizer(Answer, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # labels_att = torch.tensor(label.attention_mask).unsqueeze(0)
        # label = torch.tensor(label.input_ids).unsqueeze(0)

        if self.mode == 'train':
            item = {
                'text_input': pre_text,
                'text_output': Answer,
                'image': image,
                'answer_type': at,
                'image_name': sample['image_name'],
            }
        # some dataset don't have qid and answer_type, need to generate.
        if self.mode == 'test':
            item = {
                'text_input': pre_text,
                'text_output': Answer,
                'image': image,
                'answer_type': at,
                'image_name': sample['image_name'],
            }

        return item

    # def collate_fn_train(self, batch):
    #     input_ids = torch.stack([item['input_ids'] for item in batch])
    #     images = torch.stack([item['images'] for item in batch])
    #     labels = torch.stack([item['labels'] for item in batch])
    #     labels_att = torch.stack([item['label_att'] for item in batch])
    #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #     return {
    #         'input_ids': input_ids,
    #         'images': images,
    #         'labels': labels,
    #         'label_att': labels_att,
    #         'attention_mask': attention_mask
    #     }
    #
    # def collate_fn_test(self, batch):
    #     # ids,images,names,question, answer type, answer.
    #     input_ids = torch.stack([item['input_ids'] for item in batch])
    #     images = torch.stack([item['images'] for item in batch])
    #     image_names = [item['image_name'] for item in batch]
    #     answer_types = [item['answer_type'] for item in batch]
    #     questions = [item['question'] for item in batch]
    #     answers = [item['answer'] for item in batch]
    #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #     return {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'images': images,
    #         'images_name': image_names,
    #         'answer_type': answer_types,
    #         'question': questions,
    #         'answer': answers
    #     }


class PMC_Dataset(VQA_Dataset):
    def __init__(self, data_path, transform, img_root='',
                 seq_length=512, voc_size=32000, mode='train', answer_list_flag: bool = False):
        super().__init__(data_path, transform, mode=mode, img_root=img_root)
        self.data = pd.read_csv(os.path.join(data_path, f'{mode}.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        Question = sample['Question']
        Choice_A = sample['Choice A']
        Choice_B = sample['Choice B']
        Choice_C = sample['Choice C']
        Choice_D = sample['Choice D']
        choice_list = [Choice_A, Choice_B, Choice_C, Choice_D]
        Answer = sample['Answer']
        Answer = pre_answer(Answer)
        Question = pre_question(Question)
        ##### read image pathes #####
        img_path = os.path.join(self.data_path, self.img_root, sample['Figure_path'])
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        # Question_id = np.array(self.tokenizer(Question)['input_ids'])
        if self.mode == 'train':
            pre_text, final_o = self.random_answer(Question, Answer)

            item = {
                'text_input': pre_text,
                'text_output': Answer,
                'image': image,
            }
            return item
        if self.mode == 'test':
            Combined_choice = ''
            # random.shuffle(choice_list)
            reflect = {0: ' A:', 1: ' B:', 2: ' C:', 3: ' D:'}
            for i, choice in enumerate(choice_list):
                if Answer == choice:
                    Answer = Answer.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                        i]).replace(' D:', reflect[i])
                if Choice_A == choice:
                    Choice_A = Choice_A.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                        i]).replace(' D:', reflect[i])
                if Choice_B == choice:
                    Choice_B = Choice_B.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                        i]).replace(' D:', reflect[i])
                if Choice_C == choice:
                    Choice_C = Choice_C.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                        i]).replace(' D:', reflect[i])
                if Choice_D == choice:
                    Choice_D = Choice_D.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                        i]).replace(' D:', reflect[i])
                Combined_choice = (Combined_choice +
                                   choice.replace(' A:', reflect[i]).replace(' B:', reflect[i]).replace(' C:', reflect[
                                       i]).replace(' D:', reflect[i]))
            text = 'Question: ' + Question + ' Choices:' + Combined_choice + ' The Answer is:'
            item = {
                'question': Question,
                'text_input': text,
                'text_output': Answer,
                'image': image,
                'image_name': sample['Figure_path'],
                'label': sample['Answer_label'],
                'Choice_A': Choice_A,
                'Choice_B': Choice_B,
                'Choice_C': Choice_C,
                'Choice_D': Choice_D,
            }
            return item

class VQA2019_Dataset(VQA_Dataset):
    def __init__(self, data_path, transform, img_root='',
                 max_txt_length=512, voc_size=32000, mode='train', answer_list_flag: bool = False):
        super().__init__(data_path, transform, mode=mode, img_root=img_root, answer_list_flag= answer_list_flag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        Question = sample['question']
        Answer = sample['answer']
        Answer = pre_answer(Answer)
        at = sample['answer_type']
        Question = pre_question(Question)
        ##### read image pathes #####
        img_path = os.path.join(self.data_path, self.img_root, sample['image_name'])
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)
        if self.classifier_vqa:
            pre_text = Question
        else:
            pre_text, final_o = self.random_answer(Question, Answer)
        # final_o = self.tokenizer(pre_text, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # input_ids = final_o.input_ids
        # attention_mask = final_o.attention_mask
        # input_ids = torch.tensor(input_ids).unsqueeze(0)
        # attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        # label = self.tokenizer(Answer, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # labels_att = torch.tensor(label.attention_mask).unsqueeze(0)
        # label = torch.tensor(label.input_ids).unsqueeze(0)

        if self.mode == 'train':
            item = {
                'text_input': pre_text,
                'text_output': Answer,
                'image': image,
                'answer_type': at,
                'image_name': sample['image_name'],
            }
        # some dataset don't have qid and answer_type, need to generate.
        if self.mode == 'test':
            item = {
                'text_input': pre_text,
                'text_output': Answer,
                'image': image,
                'answer_type': at,
                'image_name': sample['image_name']
            }

        return item



def create_dataset(args):
    dataset, data_path = args.dataset_use, args.dataset_path
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_size = args.img_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # vqa_rad
    if dataset == 'radvqa':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='VQA_RAD Image Folder',
                                    answer_list_flag=args.classifier_vqa)
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='VQA_RAD Image Folder',
                                   answer_list_flag=args.classifier_vqa)


    # pathvqa
    elif dataset == 'pathvqa':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='images',
                                    answer_list_flag=args.classifier_vqa)
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='images',
                                   answer_list_flag=args.classifier_vqa)

    # slake
    elif dataset == 'slake':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='imgs',
                                    answer_list_flag=args.classifier_vqa)
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='imgs',
                                   answer_list_flag=args.classifier_vqa)


    elif dataset == 'pmcvqa':
        train_dataset = PMC_Dataset(data_path, train_transform, mode='train', img_root='images',
                                    answer_list_flag=args.classifier_vqa)
        test_dataset = PMC_Dataset(data_path, test_transform, mode='test', img_root='images',
                                   answer_list_flag=args.classifier_vqa)

    elif dataset == 'vqa2019':
        train_dataset = VQA2019_Dataset(data_path, train_transform, mode='train', img_root='images',
                                    answer_list_flag=args.classifier_vqa)
        test_dataset = VQA2019_Dataset(data_path, test_transform, mode='test', img_root='images',
                                   answer_list_flag=args.classifier_vqa)
    return train_dataset, test_dataset, ConcatDataset([train_dataset, test_dataset])




def pre_question(question):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')
    return question


def pre_answer(lanswer):
    answer = str(lanswer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', ' ')
    return answer
