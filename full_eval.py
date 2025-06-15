import torch
from transformers import AutoTokenizer, AutoModel
from stage1 import Stage1Model
from stage2 import Stage2Model
from stage3 import Stage3Model
import json
import difflib

model_path = './bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 128

id2label_tar = {0: 'O', 1: 'B-TAR', 2: 'I-TAR'}
id2label_arg = {0: 'O', 1: 'B-ARG', 2: 'I-ARG'}
group2id = {'Region': 0, 'Racism': 1, 'Sexism': 2, 'LGBTQ': 3, 'others': 4, 'non-hate': 5}
id2group = {v: k for k, v in group2id.items()}


def string_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def extract_spans(text, labels, id2label):
    spans = []
    i = 0
    while i < len(labels):
        tag = id2label.get(labels[i], 'O')
        if tag.startswith('B'):
            start = i
            i += 1
            while i < len(labels) and id2label.get(labels[i], 'O').startswith('I'):
                i += 1
            end = i
            span = text[start:end]
            if span:  # 过滤空串
                spans.append(span)
        else:
            i += 1
    return list(set(spans))


def extract_span_embeddings(text, tokenizer, encoder, spans, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = encoder(**inputs).last_hidden_state.squeeze(0)  # [seq_len, hidden]
    offset_mapping = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)["offset_mapping"]

    span_vecs = []
    for span_text in spans:
        start = text.find(span_text)
        if start == -1:
            span_vecs.append(torch.zeros(outputs.size(-1)).to(device))
            continue
        end = start + len(span_text)
        indices = [i for i, (s, e) in enumerate(offset_mapping) if s >= start and e <= end]
        if not indices:
            span_vecs.append(torch.zeros(outputs.size(-1)).to(device))
            continue
        vec = outputs[indices, :].mean(dim=0)
        span_vecs.append(vec)
    return span_vecs


def infer(text):
    model1 = Stage1Model().to(device)
    model1.load_state_dict(torch.load('./saved_model/stage1_model.pth', map_location=device))
    model1.eval()

    # 编码
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len).to(device)
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    with torch.no_grad():
        _, _, _, _, tar_pred, arg_pred = model1(input_ids, attention_mask)

    # 去掉 [CLS], [SEP], [PAD] 对应的 token 标签
    tar_spans = extract_spans(text, tar_pred, id2label_tar)
    arg_spans = extract_spans(text, arg_pred, id2label_arg)

    result1 = {
        "text": text,
        "target": tar_spans,
        "argument": arg_spans
    }

    model2 = Stage2Model().to(device)
    model2.load_state_dict(torch.load('./saved_model/stage2_model.pth', map_location=device))
    model2.eval()

    encoder = AutoModel.from_pretrained(model_path).to(device)
    encoder.eval()

    text = result1["text"]
    t_list = result1["target"]
    a_list = result1["argument"]
    result2 = []

    if not t_list:
        t_list = ["NULL"]
    if not a_list:
        a_list = ["NULL"]

    t_vecs = extract_span_embeddings(text, tokenizer, encoder, t_list, device)
    a_vecs = extract_span_embeddings(text, tokenizer, encoder, a_list, device)

    # 获取句向量（[CLS]）
    cls_inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        cls_vec = encoder(**cls_inputs).last_hidden_state[:, 0, :].squeeze(0)  # [hidden]

    for t, t_vec in zip(t_list, t_vecs):
        for a, a_vec in zip(a_list, a_vecs):
            concat_vec = torch.cat([t_vec, a_vec, cls_vec], dim=-1).unsqueeze(0)  # [1, hidden*3]
            with torch.no_grad():
                score = model2(concat_vec.to(device)).item()
            result2.append({
                "text": text,
                "target": t,
                "argument": a,
                "similarity": score
            })

    credible = [item for item in result2 if item['similarity'] > 0.6]
    for item in credible:
        item["ta_text"] = item["target"] + item["argument"]

    nms = []
    used = [False] * len(credible)

    for i, item_i in enumerate(credible):  # 非极大值抑制
        if used[i]:
            continue
        keep = True
        for j, item_j in enumerate(credible):
            if i == j or used[j]:
                continue
            sim = string_similarity(item_i['ta_text'], item_j['ta_text'])

            if sim > 0.5:
                if item_i['similarity'] >= item_j['similarity']:
                    used[j] = True  # 保留 i，舍弃 j
                else:
                    keep = False  # i 被更强的 j 覆盖
                    break

        if keep:
            nms.append(item_i)

    result2 = nms

    if not result2:
        result2.append({
                "text": text,
                "target": 'NULL',
                "argument": 'NULL',
                "similarity": 0
            })

    model3 = Stage3Model().to(device)
    model3.load_state_dict(torch.load('./saved_model/stage3_model.pth', map_location=device))
    model3.eval()

    result3 = []
    with torch.no_grad():
        for item in result2:
            text = item["text"]
            pair_text = item["target"] + " [SEP] " + item["argument"]

            # 编码
            encoded = tokenizer(
                text,
                text_pair=pair_text,
                truncation=True,
                max_length=128,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # 推理
            logits = model3(input_ids, attention_mask)
            pred_label = torch.argmax(logits, dim=1).item()

            group_name = id2group[pred_label]
            hate_label = "hate" if pred_label in [0, 1, 2, 3, 4] else "non-hate"

            result_str = f"{item['target']} | {item['argument']} | {group_name} | {hate_label}"
            result3.append(result_str)

    # 拼接最终字符串
    output_str = " [SEP] ".join(result3) + " [END]"
    return output_str


def predict(json_path):

    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析JSON数据

    # 提取所有content字段的值
    contents = [item['content'] for item in data]

    with open('output_new_0.4.txt', 'w', encoding='utf-8') as file2:
        i = 0
        for text in contents:
            result = infer(text)
            print(f"sample {i} completed")
            i += 1
            file2.write(result + '\n')


if __name__ == '__main__':
    print(infer("湖北不会有男铜的，因为都被我绞嘞"))
    # predict('data/test1.json')
