import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torchcrf import CRF


# =====================
# 1. 配置与标签映射
# =====================
model_path = "./bert-base-chinese"
max_len = 128
batch_size = 32
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map_tar = {'O': 0, 'B-TAR': 1, 'I-TAR': 2}
label_map_arg = {'O': 0, 'B-ARG': 1, 'I-ARG': 2}
group2id = {'Region': 0, 'Racism': 1, 'Sexism': 2, 'LGBTQ': 3, 'others': 4, 'non-hate':5}


# =====================
# 2. 数据预处理函数
# =====================
def parse_output(output_str):
    quads = output_str.strip().split('[SEP]')
    results = []
    for q in quads:
        parts = q.strip().replace('[END]', '').split('|')
        if len(parts) == 4:
            target, argument, group, hate = [p.strip() for p in parts]
            results.append({
                'target': target,
                'argument': argument,
                'group': group,
                'hate': hate
            })
    return results


def get_bio_labels(text, spans, label_prefix):
    labels = ['O'] * len(text)
    for span in spans:
        start = text.find(span)
        if start != -1:
            for i in range(len(span)):
                labels[start + i] = f'B-{label_prefix}' if i == 0 else f'I-{label_prefix}'
    return labels


def process_example(example):
    text = example['content']
    output = example['output']
    quads = parse_output(output)
    targets = [q['target'] for q in quads if q['target'] != 'NULL']
    arguments = [q['argument'] for q in quads if q['argument'] != 'NULL']
    bio_tar = get_bio_labels(text, targets, 'TAR')
    bio_arg = get_bio_labels(text, arguments, 'ARG')
    quads[0]['group'] = quads[0]['group'].split(", ")
    group_label = []
    for x in quads[0]['group']:
        group_label.append(group2id.get(x))
    hate_label = 1 if quads[0]['hate'] == 'hate' else 0
    return {
        'text': text,
        'bio_target': bio_tar,
        'bio_argument': bio_arg,
        'group_label': group_label,
        'hate_label': hate_label
    }


# =====================
# 3. 数据集类
# =====================
class HateDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoding = self.tokenizer(
            item['text'], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt'
        )
        tar_labels = [label_map_tar.get(l, 0) for l in item['bio_target']]
        arg_labels = [label_map_arg.get(l, 0) for l in item['bio_argument']]
        tar_labels = tar_labels[:self.max_len] + [0] * (self.max_len - len(tar_labels))
        arg_labels = arg_labels[:self.max_len] + [0] * (self.max_len - len(arg_labels))
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target_labels': torch.tensor(tar_labels),
            'arg_labels': torch.tensor(arg_labels),
            'group_label': torch.tensor(item['group_label'][0]),
            'hate_label': torch.tensor(item['hate_label'])
        }


# =====================
# 4. 模型定义
# =====================
class HateModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size

        # Token-level classifiers + CRF
        self.tar_cls = nn.Linear(hidden_size, len(label_map_tar))
        self.arg_cls = nn.Linear(hidden_size, len(label_map_arg))
        self.tar_crf = CRF(len(label_map_tar), batch_first=True)
        self.arg_crf = CRF(len(label_map_arg), batch_first=True)

        # Sequence-level classifiers
        self.group_cls = nn.Linear(hidden_size, len(group2id))
        self.hate_cls = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask,
                tar_labels=None, arg_labels=None,
                group_label=None, hate_label=None):
        # 1. BERT encoding
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = output.last_hidden_state  # (batch, seq_len, dim)
        pooled_output = output.pooler_output if hasattr(output, 'pooler_output') else seq_output[:, 0]

        # 2. Token-level logits
        tar_logits = self.tar_cls(seq_output)
        arg_logits = self.arg_cls(seq_output)

        tar_loss = arg_loss = distance_loss = None
        # 3. Compute CRF losses if labels provided
        if tar_labels is not None and arg_labels is not None:
            # CRF - negative log likelihood
            tar_nll = -self.tar_crf(tar_logits, tar_labels, mask=attention_mask.bool(), reduction='mean')
            arg_nll = -self.arg_crf(arg_logits, arg_labels, mask=attention_mask.bool(), reduction='mean')
            # Scale factors for target/argument losses
            tar_loss = tar_nll
            arg_loss = arg_nll

            # 4. Distance-based loss
            # Compute expected span centroids from true labels
            batch_size, seq_len = tar_labels.size()
            positions = torch.arange(seq_len, device=tar_labels.device).unsqueeze(0).expand(batch_size, -1)

            mask_tar = tar_labels > 0  # B or I
            mask_arg = arg_labels > 0
            # Avoid division by zero
            denom_tar = mask_tar.sum(dim=1).clamp(min=1).unsqueeze(1)
            denom_arg = mask_arg.sum(dim=1).clamp(min=1).unsqueeze(1)
            # Compute mean positions
            tar_pos = (positions * mask_tar).sum(dim=1, keepdim=True) / denom_tar
            arg_pos = (positions * mask_arg).sum(dim=1, keepdim=True) / denom_arg
            distance = torch.abs(tar_pos - arg_pos).squeeze(1)
            distance_loss = 0.1 * distance.mean()

        # 5. Sequence-level logits
        group_logits = self.group_cls(pooled_output)
        hate_logits = self.hate_cls(pooled_output)

        # 6. Return logits and all losses
        return tar_logits, arg_logits, group_logits, hate_logits, tar_loss, arg_loss, distance_loss


def train(model, dataloader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    i = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tar_labels = batch['target_labels'].to(device)
        arg_labels = batch['arg_labels'].to(device)
        group_label = batch['group_label'].to(device)
        hate_label = batch['hate_label'].to(device)

        tar_logits, arg_logits, group_logits, hate_logits, tar_loss, arg_loss, _ = model(
            input_ids, attention_mask, tar_labels, arg_labels, group_label, hate_label
        )
        group_loss = loss_fn(group_logits, group_label)
        hate_loss = loss_fn(hate_logits, hate_label)
        total_loss = tar_loss + arg_loss + group_loss + hate_loss
        total_loss.backward()
        print("batch", i, " loss:", float(total_loss))
        i += 1
        optimizer.step()
        optimizer.zero_grad()


def infer(model, tokenizer, text):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        tar_logits, arg_logits, group_logits, hate_logits, _, _, _ = model(input_ids, attention_mask)

        tar_preds = model.tar_crf.decode(tar_logits, mask=attention_mask.bool())[0]
        arg_preds = model.arg_crf.decode(arg_logits, mask=attention_mask.bool())[0]

        group_pred = torch.argmax(group_logits, dim=-1).item()

        if group_pred == 5:
            hate_pred = 0
        else:
            hate_pred = 1

    tokens = tokenizer.tokenize(text)[:len(tar_preds)]
    return {
        'text': text,
        'tokens': tokens,
        'target_labels': tar_preds,
        'argument_labels': arg_preds,
        'group': list(group2id.keys())[group_pred],
        'hateful': 'hate' if hate_pred == 1 else 'non-hate'
    }


# =====================
# 7. 运行训练
# =====================
if __name__ == '__main__':
    with open('data/train.json', encoding='utf-8') as f:
        raw_data = json.load(f)
    examples = [process_example(ex) for ex in raw_data]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = HateDataset(examples, tokenizer)
    # print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HateModel(model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        train(model, dataloader, optimizer)
        print(f"Epoch {epoch+1} completed")

    torch.save(model, './saved_model/model.pth')
    tokenizer.save_pretrained('./saved_model')
    # model = torch.load('./saved_model/model2.pth')
    # model.eval()
    # =====================
    # 8. 运行推理
    # =====================
    result = infer(model, tokenizer, "当然要反抗，但是答主说的相互歧视是不存在的。同性恋无法通过“歧视”异性恋来达到反抗的目的。因为歧视是强势群体对弱势群体发动的，单向的行为。")
    print(result)
