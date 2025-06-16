from test2 import infer, HateModel
import torch
from transformers import AutoTokenizer
import json


def extract_ordered_spans(text, labels):
    spans = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:  # B
            start = i
            i += 1
            while i < len(labels) and labels[i] == 2:
                i += 1
            end = i - 1
            span_text = ''.join(text[start:end + 1])
            spans.append((start, end, span_text))
        else:
            i += 1
    return spans


def convert_to_quadruple(sample):
    text = sample['text']
    target_labels = sample['target_labels']
    argument_labels = sample['argument_labels']
    group = sample['group']
    hateful = sample['hateful']

    target = extract_ordered_spans(text, target_labels)
    argument = extract_ordered_spans(text, argument_labels)

    if not target:
        target1 = 'NULL'
    else:
        target1 = target[0][2]
    if not argument:
        argument1 = 'NULL'
    else:
        argument1 = argument[0][2]

    return f"{target1} | {argument1} | {group} | {hateful} [END]"


def predict(model, tokenizer, json_path):

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析JSON数据

    # 提取所有content字段的值
    contents = [item['content'] for item in data]

    with open('output7.txt', 'w', encoding='utf-8') as file2:
        for text in contents:
            result = infer(model, tokenizer, text)
            result2 = convert_to_quadruple(result)
            file2.write(result2 + '\n')


if __name__ == '__main__':
    model = torch.load('./saved_model/model2.pth')
    tokenizer = AutoTokenizer.from_pretrained('./saved_model')
    model.eval()
    result = infer(model, tokenizer,
                   "当然要反抗，但是答主说的相互歧视是不存在的。同性恋无法通过“歧视”异性恋来达到反抗的目的。因为歧视是强势群体对弱势群体发动的，单向的行为。")
    # predict(model, tokenizer, 'data/test1.json')
    print(convert_to_quadruple(result))


