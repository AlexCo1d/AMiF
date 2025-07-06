import difflib
import os
import argparse
import re
import json

# import Levenshtein

contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                "couldnt": "couldn't",
                "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't",
                "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                "hed": "he'd", "hed've": "he'd've",
                "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
                "Id've": "I'd've", "I'dve": "I'd've",
                "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
                "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
                "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
                "mightn'tve": "mightn't've", "mightve": "might've",
                "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                "oclock": "o'clock", "oughtnt": "oughtn't",
                "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
                "shed've": "she'd've", "she'dve": "she'd've",
                "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
                "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
                "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
                "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've",
                "someone'dve": "someone'd've",
                "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
                "somethingd've": "something'd've",
                "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                "thered": "there'd", "thered've": "there'd've",
                "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd",
                "theyd've": "they'd've",
                "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                "twas": "'twas", "wasnt": "wasn't",
                "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
                "whatll": "what'll", "whatre": "what're",
                "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                "wheres": "where's", "whereve": "where've",
                "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
                "whos": "who's", "whove": "who've", "whyll": "why'll",
                "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
                "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                "yall'd've": "y'all'd've",
                "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
                "youd've": "you'd've", "you'dve": "you'd've",
                "youll": "you'll", "youre": "you're", "youve": "you've"}

manualMap = {'none': '0',
             'zero': '0',
             'one': '1',
             'two': '2',
             'three': '3',
             'four': '4',
             'five': '5',
             'six': '6',
             'seven': '7',
             'eight': '8',
             'nine': '9',
             'ten': '10'
             }
articles = ['a',
            'an',
            'the'
            ]
punct = [';', r"/", '[', ']', '"', '{', '}',
         '(', ')', '=', '+', '\\', '_', '-',
         '>', '<', '@', '`', ',', '?', '!']
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(\d)(,)(\d)")


def compute_vqa_acc(vqa_results: [], epoch=0, args=None, dataloader=None):
    """
    :param vqa_results: list : 'image_name':image_names[idx], #获取图片名
                                "question": questions[idx],  # 当前问题
                                "pred": pred_answer,  # 预测的答案
                                "answer": answers[idx],  # 实际答案
                                "answer_type": answer_types[idx]  # 答案类型
    :param args: args maybe useful
    :return: acc: {
                    'epoch': epoch,
                    'overall': overall,
                    'OPEN': OPEN,
                    'CLOSED': CLOSED
                    }
    """
    print('vqa_results:', len(vqa_results))
    if args is None:
        args = {}
    open_list = []
    closed_list = []
    if dataloader is not None:
        answer_list = dataloader.dataset.answer_list
    for item in vqa_results:
        gt = item['answer']
        pred = item['pred']
        type = item['answer_type']
        pred = pred.replace('\n', ' ')
        pred = pred.replace('\t', ' ')
        pred = pred.strip()
        pred = pre_answer(pred)
        gt = pre_answer(gt)
        if type == 'OPEN':
            # print('pred:', pred, ' gt:', gt, ' sim:', sim)
            open_list.append(int(pred == gt))
        else:
            closed_list.append(int(gt == pred))

    acc = {
        'epoch': epoch,
        'overall': (sum(open_list) + sum(closed_list)) / len(vqa_results),
        'OPEN': sum(open_list) / len(open_list),
        'CLOSED': sum(closed_list) / len(closed_list)
    }
    return acc


def runtime_vqa_acc(gt: [], pre: []):
    assert len(gt) == len(pre)
    open_list = []
    closed_list = []
    for g, p in zip(gt, pre):
        p = p.replace('\n', ' ')
        p = p.replace('\t', ' ')
        p = p.strip()
        p = pre_answer(p)
        g = pre_answer(g)
        if g == 'yes' or g == 'no':
            closed_list.append(int(p == g))
        else:
            open_list.append(int(g == p))
    acc = {
        'overall': (sum(open_list) + sum(closed_list)) / len(pre),
        'OPEN': sum(open_list) / len(open_list),
        'CLOSED': sum(closed_list) / len(closed_list)
    }
    return acc


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("",
                              outText,
                              re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText


def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = processPunctuation(answer)
    answer = processDigitArticle(answer)
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-').replace('-', '')
    return answer


# def get_most_similar(answer_list, pred):
#     most_similar_str = None
#     most_similar_index = None
#     highest_similarity = -1
#     for i, s in enumerate(answer_list):
#         similarity = Levenshtein.ratio(s, pred)
#         if similarity > highest_similarity:
#             most_similar_str = str(s)
#             most_similar_index = i
#             highest_similarity = similarity
#     return most_similar_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="")
    args = parser.parse_args()
    json_file = json.load(open(args.file, 'r'))
    print(compute_vqa_acc(json_file))


if __name__ == '__main__':
    main()
