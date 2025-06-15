import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from torch.optim import AdamW


# ----- 配置 -----
group2id = {'Region': 0, 'Racism': 1, 'Sexism': 2, 'LGBTQ': 3, 'others': 4, 'non-hate': 5}
model_path = "./chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 1
batch_size = 32


# ----- 数据处理函数 -----
def build_group_classification_dataset(raw_data, group2id):
    dataset = []

    def parse_quads(output_str, has_score=False):
        parts = output_str.strip().replace("[END]", "").split("[SEP]")
        results = []
        for p in parts:
            fields = [x.strip() for x in p.strip().split("|")]
            if has_score and len(fields) == 5:
                t, a, group, hate, score = fields
                results.append((t, a, group, float(score)))
            elif not has_score and len(fields) == 4:
                t, a, group, hate = fields
                results.append((t, a, group, 1.0))  # 真实样本权重设为1.0
        return results

    for ex in raw_data:
        text = ex["content"]

        for t, a, g, w in parse_quads(ex["output"], has_score=False):
            if not t or not a or g not in group2id:
                continue
            dataset.append({
                "text": text,
                "target": t,
                "argument": a,
                "label": group2id[g],
                "weight": w
            })

        if "extra" in ex:
            for t, a, g, w in parse_quads(ex["extra"], has_score=True):
                if not t or not a or g not in group2id:
                    continue
                dataset.append({
                    "text": text,
                    "target": t,
                    "argument": a,
                    "label": group2id[g],
                    "weight": w
                })

    return dataset


# ----- Dataset 类 -----
class GroupClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["text"],
            text_pair=item["target"] + " [SEP] " + item["argument"],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": item["label"],
            "weight": item["weight"]
        }


# ----- 模型类 -----
class Stage3Model(nn.Module):
    def __init__(self, pretrained_model="chinese-roberta-wwm-ext-large", num_classes=6):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits


# ----- 训练函数 -----
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    i = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        weights = batch["weight"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        weighted_loss = (loss * weights).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        print(f"Epoch {epoch + 1} | Batch {i} | Loss: {weighted_loss.item():.4f}")
        i += 1


# ----- 评估函数 -----
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            pred_labels = torch.argmax(logits, dim=1)

            preds.extend(pred_labels.cpu().numpy())
            labels.extend(label_ids.cpu().numpy())

    print(classification_report(labels, preds, target_names=list(group2id.keys())))


if __name__ == "__main__":

    with open("data/train_with_extra.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    dataset = build_group_classification_dataset(raw_data, group2id)
    train_dataset = GroupClassificationDataset(dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Stage3Model().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        evaluate(model, train_loader, device)

    torch.save(model.state_dict(), 'saved_model/stage3_model.pth')
