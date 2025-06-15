import json
import torch
from transformers import AutoTokenizer, AutoModel


def extract_span_embeddings(text, tokenizer, model, spans):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.squeeze(0)  # [seq_len, hidden_size]
    offset_map = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=128)['offset_mapping']
    span_vecs = []
    for span in spans:
        start = text.find(span)
        end = start + len(span)
        indices = [i for i, (s, e) in enumerate(offset_map) if s >= start and e <= end]
        if not indices:
            span_vecs.append(torch.zeros(outputs.size(-1)).to(model.device))
            continue
        vec = outputs[indices].mean(dim=0)
        span_vecs.append(vec)
    return span_vecs


def parse_output(output_str):
    quads = output_str.strip().split('[SEP]')
    results = []
    for q in quads:
        parts = q.strip().replace('[END]', '').split('|')
        if len(parts) == 4:
            target, argument, group, hate = [p.strip() for p in parts]
            results.append((target, argument, group, hate))
    return results


def generate_augmented_sample(example, tokenizer, model, neg_num=4, coincide_threshold=0.7):
    import random
    from sklearn.metrics.pairwise import cosine_similarity

    def overlap_ratio(s1, s2):
        if not s1 or not s2:
            return 0
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / max(len(set1), len(set2))

    text = example["content"]
    gold_quads = parse_output(example["output"])
    gold_pairs = [(t, a) for t, a, _, _ in gold_quads if t and a]
    gold_vecs = [(extract_span_embeddings(text, tokenizer, model, [t])[0],
                  extract_span_embeddings(text, tokenizer, model, [a])[0])
                 for t, a in gold_pairs]

    # 伪样本构造
    text_len = len(text)
    used_targets = set(t for t, _ in gold_pairs)
    used_arguments = set(a for _, a in gold_pairs)
    generated = []

    for _ in range(neg_num):
        if not gold_pairs:
            continue
        ref_t, ref_a, ref_g, ref_h = random.choice(gold_quads)
        if not ref_t or not ref_a:
            continue
        len_t, len_a = len(ref_t), len(ref_a)

        max_attempts = 30
        for _ in range(max_attempts):
            start_t = random.randint(0, max(0, text_len - len_t))
            start_a = random.randint(0, max(0, text_len - len_a))
            cand_t = text[start_t:start_t + len_t]
            cand_a = text[start_a:start_a + len_a]

            if not cand_t or not cand_a:
                continue
            if cand_t in used_targets or cand_a in used_arguments:
                continue
            overlap_ok = all(overlap_ratio(cand_t, gt) <= coincide_threshold and
                             overlap_ratio(cand_a, ga) <= coincide_threshold
                             for gt, ga in gold_pairs)
            if not overlap_ok:
                continue

            t_vec = extract_span_embeddings(text, tokenizer, model, [cand_t])[0]
            a_vec = extract_span_embeddings(text, tokenizer, model, [cand_a])[0]
            pred_vec = torch.cat([t_vec, a_vec], dim=-1).unsqueeze(0)

            sims = [cosine_similarity(pred_vec.cpu().numpy(),
                                      torch.cat([tv, av]).unsqueeze(0).cpu().numpy())[0][0]
                    for tv, av in gold_vecs]
            sim_label = max(sims) if sims else 0.0

            generated.append(f"{cand_t} | {cand_a} | {ref_g} | {ref_h} | {sim_label:.4f}")
            used_targets.add(cand_t)
            used_arguments.add(cand_a)
            break

    # 构造输出
    return {
        "id": example["id"],
        "content": text,
        "output": example["output"],
        "extra": "[SEP] ".join(generated) + " [END]" if generated else ""
    }


if __name__ == '__main__':
    with open("data/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese").to(device)
    model.eval()

    augmented_data = []
    i = 0
    for example in data:
        augmented = generate_augmented_sample(example, tokenizer, model)
        augmented_data.append(augmented)
        print(i, "completed")
        i += 1

    # 最终写入
    with open("data/train_with_extra.json", "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
