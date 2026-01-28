import argparse, json, os
from collections import Counter
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, BertForSequenceClassification, TrainingArguments,
    Trainer, BertConfig
)
from tokenizers import BertWordPieceTokenizer

import evaluate


def train_biomedical_tokenizer(corpus_path: str, vocab_size: int = 30000, lowercase: bool = True, min_frequency: int = 3):
    trainer = BertWordPieceTokenizer(lowercase=lowercase)
    trainer.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    return trainer

def select_domain_tokens(domain_tokenizer, base_tokenizer, corpus_path: str, k: int = 5000):
    base_vocab = set(base_tokenizer.vocab.keys())
    counter = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Counting token frequency"):
            text = json.loads(line).get("text", line.strip())
            tokens = domain_tokenizer.encode(text).tokens
            for tok in tokens:
                if tok not in base_vocab:
                    counter[tok] += 1
    most_common = [tok for tok, _ in counter.most_common(k)]
    print(f"Selected {len(most_common)} domain-specific tokens")
    return most_common

def expand_bert_tokenizer_and_model(base_model_name: str, new_tokens: list, output_dir: str, num_labels=10):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = BertForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    added = tokenizer.add_tokens(new_tokens)
    assert added == len(new_tokens), "Some tokens duplicated with base vocab!"
    model.resize_token_embeddings(len(tokenizer))
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    print(f"Expanded tokenizer & model saved to: {output_dir}")
    return tokenizer, model

def train_on_hoc(tokenizer, model, output_dir, max_length=128, num_labels=11):
    from datasets import load_dataset

    # 加载 parquet 数据
    dataset = load_dataset('parquet', data_files={
        'train': 'train.parquet',
        'test': 'test.parquet'
    })

    # 自动划分 validation
    split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
    dataset['train'] = split_dataset['train']
    dataset['validation'] = split_dataset['test']

    # 预处理函数
    def preprocess(example):
        res = tokenizer(example['text'], truncation=True, padding="max_length", max_length=max_length)
        res["labels"] = example["label"]
        return res

    dataset = dataset.map(preprocess, batched=False)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    # 训练参数
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        learning_rate=2e-5,
        logging_steps=20,
        save_total_limit=1,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
    )

    # ==== 新增评估指标 ====
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(-1)
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        accuracy = acc_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        return {"accuracy": accuracy, "macro_f1": f1}


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print("Fine-tuning finished. Model saved.")

    # ==== 打印验证集/测试集分数 ====
    val_result = trainer.evaluate()
    print("Validation set results:", val_result)

    test_result = trainer.evaluate(eval_dataset=dataset["test"])
    print("Test set results:", test_result)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",  default="pubmed_sampled_corpus.jsonline")
    ap.add_argument("--vocab_size", type=int, default=30000, help="biomedical tokenizer 词表大小")
    ap.add_argument("--num_new_tokens", type=int, default=5000, help="要并入的领域 token 数")
    ap.add_argument("--output_dir", type=str, default="expanded_bert", help="保存扩展后 tokenizer+model 的文件夹")
    ap.add_argument("--base_model", type=str, default="bert-base-uncased", help="原始 BERT 模型/分词器名称")
    ap.add_argument("--num_labels", type=int, default=11, help="HoC 任务类别数")
    return ap.parse_args()

def main():
    args = parse_args()
    print("Step 1: Training biomedical WordPiece tokenizer ...")
    #domain_tokenizer = train_biomedical_tokenizer(args.corpus, vocab_size=args.vocab_size)
    print("Step 2: Selecting domain-specific tokens ...")
    #base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    #new_tokens = select_domain_tokens(domain_tokenizer, base_tokenizer, args.corpus, k=args.num_new_tokens)
    print("Step 3: Expanding tokenizer & model ...")
    #tokenizer, model = expand_bert_tokenizer_and_model(args.base_model, new_tokens, args.output_dir, num_labels=args.num_labels)
    print("Step 4: Fine-tuning on HoC ...")


    tokenizer = AutoTokenizer.from_pretrained("expanded_bert")
    model = BertForSequenceClassification.from_pretrained("expanded_bert")
    # ========= Report 原始数据收集 =========
    from pathlib import Path, PurePath
    import numpy as np, json

    EXP_DIR = Path("expanded_bert")              # 你的平铺文件夹
    BASE_TOK = AutoTokenizer.from_pretrained(args.base_model)

# (1) 领域词表大小（行数=token 数）
    vocab_size = sum(1 for _ in open(EXP_DIR / "vocab.txt", "r", encoding="utf-8"))
    print(f"[Report] Domain tokenizer vocab size : {vocab_size}")

# (2) 新增 token 清单（先看前 30 个）
    with open(EXP_DIR / "added_tokens.json", encoding="utf-8") as fh:
        new_tokens = json.load(fh)
        if isinstance(new_tokens, dict):          # ← 兼容旧格式
            new_tokens = list(new_tokens.keys())

    print(f"[Report] First 30 new tokens : {new_tokens[:30]}")

# (3) 三句样例在 base / expanded 下的长度比较
    samples = [
    "Peking University is located in Haidian district.",
    "Computer Science is the flagship major of Peking University.",
    "The School of Electronic Engineering and Computer Science enrolls approximately five hundred new students each year.",
    ]
    lens_base     = [len(BASE_TOK(s)['input_ids']) for s in samples]
    lens_expanded = [len(tokenizer(s)['input_ids']) for s in samples]
    print(f"[Report] Lengths (base)     : {lens_base},  avg = {np.mean(lens_base):.1f}")
    print(f"[Report] Lengths (expanded) : {lens_expanded},  avg = {np.mean(lens_expanded):.1f}")

# (4) 新增参数量 = 新 token × hidden_size
    hidden_size   = model.config.hidden_size          # 通常 768
    added_params  = len(new_tokens) * hidden_size
    print(f"[Report] Added tokens = {len(new_tokens)},  new embedding params = {added_params:,}")
# ========= End Report part =========

    train_on_hoc(tokenizer, model, args.output_dir, num_labels=args.num_labels)

if __name__ == "__main__":
    main()
