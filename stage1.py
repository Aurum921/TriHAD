# stage1_train.py
# 第一阶段：训练target和argument的BIO标签预测模型（带CRF）

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from torch.optim import AdamW

model_path = './chinese-roberta-wwm-ext-large'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_len = 128
batch_size = 32
epochs = 10

label_map_tar = {'O': 0, 'B-TAR': 1, 'I-TAR': 2}
label_map_arg = {'O': 0, 'B-ARG': 1, 'I-ARG': 2}


class Stage1Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            text = item['content']
            quads = item['output'].strip().split('[SEP]')
            targets, arguments = set(), set()
            for q in quads:
                q = q.replace('[END]', '').strip()
                if not q: continue
                parts = q.split('|')
                if len(parts) >= 2:
                    targets.add(parts[0].strip())
                    arguments.add(parts[1].strip())
            tar_labels = ['O'] * len(text)
            arg_labels = ['O'] * len(text)
            for t in targets:
                start = text.find(t)
                if start != -1:
                    for i in range(len(t)):
                        tar_labels[start + i] = 'B-TAR' if i == 0 else 'I-TAR'
            for a in arguments:
                start = text.find(a)
                if start != -1:
                    for i in range(len(a)):
                        arg_labels[start + i] = 'B-ARG' if i == 0 else 'I-ARG'
            self.samples.append((text, tar_labels, arg_labels))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, tar, arg = self.samples[idx]
        enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze()
        attention_mask = enc['attention_mask'].squeeze()
        tar_labels = [label_map_tar.get(x, 0) for x in tar][:max_len] + [0] * (max_len - len(tar))
        arg_labels = [label_map_arg.get(x, 0) for x in arg][:max_len] + [0] * (max_len - len(arg))
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tar_labels': torch.tensor(tar_labels),
            'arg_labels': torch.tensor(arg_labels)
        }


class Stage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.tar_fc = nn.Linear(hidden, len(label_map_tar))
        self.arg_fc = nn.Linear(hidden, len(label_map_arg))
        self.tar_crf = CRF(len(label_map_tar), batch_first=True)
        self.arg_crf = CRF(len(label_map_arg), batch_first=True)

    def forward(self, input_ids, attention_mask, tar_labels=None, arg_labels=None):
        out = self.bert(input_ids, attention_mask).last_hidden_state
        tar_logits = self.tar_fc(out)
        arg_logits = self.arg_fc(out)
        tar_loss, arg_loss = None, None
        if tar_labels is not None:
            tar_loss = -self.tar_crf(tar_logits, tar_labels, mask=attention_mask.bool(), reduction='mean')
        if arg_labels is not None:
            arg_loss = -self.arg_crf(arg_logits, arg_labels, mask=attention_mask.bool(), reduction='mean')
        tar_preds = self.tar_crf.decode(tar_logits, mask=attention_mask.bool())
        arg_preds = self.arg_crf.decode(arg_logits, mask=attention_mask.bool())
        return tar_logits, arg_logits, tar_loss, arg_loss, tar_preds[0], arg_preds[0]


def train_stage1():
    with open('data/train.json', encoding='utf-8') as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = Stage1Dataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Stage1Model().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        model.train()
        i = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            tar = batch['tar_labels'].to(device)
            arg = batch['arg_labels'].to(device)
            _, _, tar_loss, arg_loss, _, _ = model(input_ids, mask, tar, arg)
            loss = tar_loss + arg_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch + 1} : batch {i} loss {float(loss)}")
            i += 1
    torch.save(model.state_dict(), 'saved_model/stage1_model_roberta.pth')
    tokenizer.save_pretrained('saved_model')
    return model


if __name__ == '__main__':
    model = train_stage1()

