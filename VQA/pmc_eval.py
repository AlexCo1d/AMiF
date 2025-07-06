import difflib
import os
import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import utils
import Utils.misc as misc
import Utils.lr_sched as lr_sched
from Dataset import create_dataset
from model.Former_T5 import Former_T5
from vqaTools.vqaEvaluate import compute_vqa_acc


def evaluation_pmc(model, data_loader, device, args):
    # test
    if args.distributed:
        t_model = model.module
        t_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate PMC_VQA test result:'
    print_freq = 10
    ACC = 0
    cc = 0
    result = []

    for n, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        text_output = t_model.predict_answers(b, dataloader=data_loader)
        for idx, answer in enumerate(text_output):
            result.append({
                'image_name': b['image_name'][idx],
                'question': b['question'][idx],
                'pred': answer,
                'answer': b['text_output'][idx],
                'label': b['label'][idx],
            })
            Choice_A = b['Choice_A'][idx]
            Choice_B = b['Choice_B'][idx]
            Choice_C = b['Choice_C'][idx]
            Choice_D = b['Choice_D'][idx]
            Choice_list = [Choice_A, Choice_B, Choice_C, Choice_D]
            index_pred = find_most_similar_index(Choice_list, answer)
            index_label = find_most_similar_index(Choice_list, b['label'][idx])
            if index_pred == index_label:
                ACC = ACC + 1
            cc = cc + 1

    json.dump(result,
              open(os.path.join(args.result_dir, 'vqa_result_%s.json' % (args.dataset_use)), 'w'))

    acc = ACC / cc

    print(f'{args.dataset_use} acc: {acc}')


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    # Return the index of the most similar string
    return most_similar_index

