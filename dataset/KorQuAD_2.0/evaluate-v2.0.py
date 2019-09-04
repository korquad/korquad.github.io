from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
from bs4 import BeautifulSoup

'''KorQuAD v1.0에 대한 공식 평가 스크립트 '''
'''본 스크립트는 SQuAD v1.1 평가 스크립트 https://rajpurkar.github.io/SQuAD-explorer/ 를 바탕으로 작성됨.'''

def normalize_answer(s):  
    
    def remove_html(text):
        return BeautifulSoup(text, features="html5lib").get_text()
    
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)      
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(remove_html(s)))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
#     print("PREDICTION:", " ".join(prediction_tokens))
#     print("GT:", " ".join(ground_truth_tokens))
    
   
    #F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)
        
    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)   
        
    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    dict_ = {}
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa["type"] in dict_.keys():
                    dict_[qa["type"]]["cnt"] += 1
                else:
                    dict_[qa["type"]] = {"cnt":1,"em":0.0,"f1":0.0}
                    
                if str(qa['id']) not in predictions.keys():
                    message = 'Unanswered question ' + str(qa['id'])+ \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[str(qa['id'])]
                
                em_ = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1_ = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

                dict_[qa["type"]]["em"] += em_
                dict_[qa["type"]]["f1"] += f1_
                    
                
                exact_match += em_
                f1 += f1_

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    
    print("--------------------------------------------")
    print("{:12} | EM = {:.4f}% | F1 = {:.4f}%".format("Total",exact_match, f1))
    print("{:12} | EM = {:.4f}% | F1 = {:.4f}%".format("Long Type",100.0*dict_["long"]["em"]/dict_["long"]["cnt"], 100.0*dict_["long"]["f1"]/dict_["long"]["cnt"]))
    print("{:12} | EM = {:.4f}% | F1 = {:.4f}%".format("Short Type",100.0*dict_["short"]["em"]/dict_["short"]["cnt"], 100.0*dict_["short"]["f1"]/dict_["short"]["cnt"]))
    print("--------------------------------------------\n")
    
    
    return {'exact_match': exact_match, 'f1': f1}


if __name__ == '__main__':
    expected_version = 'KorQuAD_v2.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for KorQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    file_names = os.listdir(args.dataset_file)
    file_names = [a for a in file_names if a[-4:]=="json"]
    dataset = []
    for file_name in file_names:
        data_file = os.path.join(args.dataset_file, file_name)
        with open(data_file) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset.extend(dataset_json['data'])
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
