import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
import random

# =====================
# 配置
# =====================
model_path = "./chinese-roberta-wwm-ext-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128
batch_size = 32
epochs = 3


# =====================
# 工具函数
# =====================
def parse_output(output_str):
    quads = output_str.strip().split('[SEP]')
    results = []
    for q in quads:
        parts = q.strip().replace('[END]', '').split('|')
        if len(parts) == 4:
            target, argument, _, _ = [p.strip() for p in parts]
            results.append((target, argument))
    return results


def extract_span_embeddings(text, tokenizer, model, spans):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.squeeze(0)  # [seq_len, hidden_size]
    char_to_tok_map = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)['offset_mapping']

    span_vecs = []
    for span_text in spans:
        start = text.find(span_text)
        if start == -1:
            span_vecs.append(torch.zeros(outputs.size(-1)).to(device))
            continue
        end = start + len(span_text)
        indices = [i for i, (s, e) in enumerate(char_to_tok_map) if s >= start and e <= end]
        if not indices:
            span_vecs.append(torch.zeros(outputs.size(-1)).to(device))
            continue
        vec = outputs[indices, :].mean(dim=0)
        span_vecs.append(vec)
    return span_vecs


def build_pairs(text, gold_pairs, model, tokenizer, neg_num=4, len_range=0.4, coincide_range=0.7):
    # 负样本与真实样本有不超过0.7的重合，长度相差不超过0.4
    # 去重后的 gold target / argument
    targets = list({t for t, _ in gold_pairs})
    arguments = list({a for _, a in gold_pairs})
    all_pairs = [(t, a) for t in targets for a in arguments]

    gold_vecs = [
        (extract_span_embeddings(text, tokenizer, model, [t])[0],
         extract_span_embeddings(text, tokenizer, model, [a])[0])
        for t, a in gold_pairs
    ]

    result = []

    # 正样本（真实对）
    for t, a in all_pairs:
        t_vec = extract_span_embeddings(text, tokenizer, model, [t])[0]
        a_vec = extract_span_embeddings(text, tokenizer, model, [a])[0]
        pred_vec = torch.cat([t_vec, a_vec], dim=-1).unsqueeze(0)
        sims = [
            cosine_similarity(pred_vec.cpu().numpy(),
                              torch.cat([tv, av]).unsqueeze(0).cpu().numpy())[0][0]
            for tv, av in gold_vecs
        ]
        sim_label = max(sims) if sims else 0.0
        result.append({"target": t, "argument": a, "similarity": sim_label})

    # 伪样本（近似负对）
    used_pairs = set((t, a) for t, a in all_pairs)
    used_targets = set(t for t, _ in used_pairs)
    used_arguments = set(a for _, a in used_pairs)

    def overlap_ratio(s1, s2):
        if not s1 or not s2:
            return 0
        l1, l2 = len(s1), len(s2)
        common = len(set(s1) & set(s2))
        return common / max(l1, l2)

    text_len = len(text)

    for _ in range(neg_num):
        if not gold_pairs:
            continue

        # 1. 随机选一个真实 ta 对作为参考长度
        ref_t, ref_a = random.choice(gold_pairs)
        len_t = len(ref_t)
        len_a = len(ref_a)

        def valid_span(start, length):
            return text[start:start + length] if 0 <= start < text_len and start + length <= text_len else None

        max_attempts = 30
        rand_t = rand_a = None
        found = False
        for _ in range(max_attempts):
            # 2. 生成长度限制内的新片段
            len_t_new = max(1, int(len_t * (1 + random.uniform(-len_range, len_range))))
            len_a_new = max(1, int(len_a * (1 + random.uniform(-len_range, len_range))))

            start_t = random.randint(0, max(text_len - len_t_new, 0))
            start_a = random.randint(0, max(text_len - len_a_new, 0))
            rand_t = valid_span(start_t, len_t_new)
            rand_a = valid_span(start_a, len_a_new)
            if not rand_t or not rand_a:
                continue

            # 3. 判重 + 重叠度过滤
            if rand_t in used_targets or rand_a in used_arguments:
                continue

            overlap_ok = all(overlap_ratio(rand_t, gt) <= coincide_range and
                             overlap_ratio(rand_a, ga) <= coincide_range
                             for gt, ga in gold_pairs)
            if not overlap_ok:
                continue

            used_targets.add(rand_t)
            used_arguments.add(rand_a)
            found = True
            break

        if not found:
            continue

        # 4. 向量计算 + 余弦最大相似度作为 soft label
        t_vec = extract_span_embeddings(text, tokenizer, model, [rand_t])[0]
        a_vec = extract_span_embeddings(text, tokenizer, model, [rand_a])[0]
        pred_vec = torch.cat([t_vec, a_vec], dim=-1).unsqueeze(0)

        sims = [
            cosine_similarity(pred_vec.cpu().numpy(),
                              torch.cat([tv, av]).unsqueeze(0).cpu().numpy())[0][0]
            for tv, av in gold_vecs
        ]
        sim_label = max(sims) if sims else 0.0

        result.append({"target": rand_t, "argument": rand_a, "similarity": sim_label})

    return result


# =====================
# 数据集
# =====================
class Stage2Dataset(Dataset):
    def __init__(self, data, tokenizer, encoder):
        self.samples = []
        self.tokenizer = tokenizer
        self.encoder = encoder
        for ex in data:
            text = ex['content']
            gold_pairs = parse_output(ex['output'])
            sim_pairs = build_pairs(text, gold_pairs, encoder, tokenizer)
            for item in sim_pairs:
                self.samples.append({
                    'text': text,
                    'target': item['target'],
                    'argument': item['argument'],
                    'similarity': item['similarity']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        inputs = self.tokenizer(
            item['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=max_len
        )
        t_vec = extract_span_embeddings(item['text'], self.tokenizer, self.encoder, [item['target']])[0]
        a_vec = extract_span_embeddings(item['text'], self.tokenizer, self.encoder, [item['argument']])[0]

        with torch.no_grad():
            cls_vec = self.encoder(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[:, 0, :].squeeze(0)

        return {
            'ta_vec': torch.cat([t_vec, a_vec, cls_vec]),
            'sim_label': torch.tensor(item['similarity'], dtype=torch.float)
        }


# =====================
# 模型
# =====================
class Stage2Model(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, ta_vec):
        return self.regressor(ta_vec).squeeze(1)


# =====================
# 训练主函数
# =====================
def train():
    with open('data/train.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = AutoModel.from_pretrained(model_path).to(device)
    encoder.eval()  # 冻结参数提特征

    dataset = Stage2Dataset(raw_data, tokenizer, encoder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Stage2Model(hidden_size=encoder.config.hidden_size).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(dataloader):
            ta_vec = batch['ta_vec'].to(device)
            sim_label = batch['sim_label'].to(device)
            pred_score = model(ta_vec)
            loss = loss_fn(pred_score, sim_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch + 1} | Batch {i} | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'saved_model/stage2_model2.pth')


if __name__ == '__main__':
    train()
