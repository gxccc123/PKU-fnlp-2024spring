#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py  —— 统一脚本：Log‑Linear  与  BERT 文本分类
Author: (your name)
Course: Foundations of NLP, PKU 2025 Spring

Usage
-----
# 单模型单数据集
python main.py --model loglinear --dataset 20news
python main.py --model loglinear --dataset hoc
python main.py --model bert       --dataset 20news
python main.py --model bert       --dataset hoc

# 一键全部实验
python main.py --all
"""
import argparse, json, os, random, pathlib, sys
import numpy as np, pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import datasets
# ---------------- 全局随机种子 ----------------
RNG = 42
random.seed(RNG); np.random.seed(RNG)

# ================= 工具函数 ===================
def compute_metrics(y_true, y_pred):
    """返回 dict: accuracy / macro_f1 / micro_f1"""
    from sklearn.metrics import accuracy_score, f1_score
    acc   = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average='macro')
    micro = f1_score(y_true, y_pred, average='micro')
    return {"accuracy": round(acc,4), "macro_f1": round(macro,4), "micro_f1": round(micro,4)}

def save_results(json_path, model_key, dataset_key, metrics_train, metrics_test):
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault(model_key, {})
    data[model_key][dataset_key] = {"train": metrics_train, "test": metrics_test}
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

# ================= 数据加载 ===================
def load_20news():
    from datasets import load_dataset
    ds = load_dataset('SetFit/20_newsgroups')
    return ds['train']['text'], ds['train']['label'], ds['test']['text'], ds['test']['label'], 20


def load_hoc():
    def read(split):
        df = pd.read_parquet(f'./data/HoC/{split}.parquet')
        return df['text'].tolist(), df['label'].tolist()
    X_train, y_train = read('train')
    X_test,  y_test  = read('test')
    return X_train, y_train, X_test, y_test, 11

# ================= Log‑Linear =================
def run_loglinear(X_train, y_train, X_test, y_test):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1,2),
            max_features=50000,
            sublinear_tf=True,
            stop_words='english')),
        ('lr', LogisticRegression(
            max_iter=1000,
            C=4.0,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RNG))
    ])
    clf.fit(X_train, y_train)
    return clf.predict(X_train), clf.predict(X_test)

# ===================  BERT  ===================
def run_bert(X_train, y_train, X_test, y_test, num_labels):
    from transformers import (BertTokenizerFast, BertForSequenceClassification,
                              TrainingArguments, Trainer)
    # 构造 HuggingFace Dataset
    def to_dataset(texts, labels):
        return Dataset.from_dict({'text': texts, 'labels': labels})
    ds = DatasetDict({
        'train': to_dataset(X_train, y_train),
        'test' : to_dataset(X_test,  y_test)
    })

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ds = ds.map(lambda x: tokenizer(x['text'],
                                    truncation=True,
                                    padding="max_length",
                                    max_length=128), batched=True)
    ds.set_format(type='torch', columns=['input_ids','token_type_ids',
                                         'attention_mask','labels'])

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_labels)

    training_args = TrainingArguments(
    output_dir='outputs/tmp',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=200,
    seed=RNG
)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=ds['train'])

    trainer.train()

    # 预测
    def predict(split):
        logits = trainer.predict(ds[split]).predictions
        return np.argmax(logits, axis=1)
    return predict('train'), predict('test')

# ================ 主流程 ======================
def experiment(model_name, dataset_name):
    # 读数据
    if dataset_name == '20news':
        X_tr, y_tr, X_te, y_te, num_labels = load_20news()
    else:
        X_tr, y_tr, X_te, y_te, num_labels = load_hoc()

    # 跑模型
    if model_name == 'loglinear':
        yhat_tr, yhat_te = run_loglinear(X_tr, y_tr, X_te, y_te)
    else:
        yhat_tr, yhat_te = run_bert(X_tr, y_tr, X_te, y_te, num_labels)

    # 计算 & 保存
    mtrain = compute_metrics(y_tr, yhat_tr)
    mtest  = compute_metrics(y_te, yhat_te)
    save_results('outputs/results.json', model_name, dataset_name,
                 mtrain, mtest)

    print(f"✔ {model_name.upper()} on {dataset_name}:")
    print("  train =", mtrain)
    print("  test  =", mtest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   choices=['loglinear','bert'])
    parser.add_argument('--dataset', choices=['20news','hoc'])
    parser.add_argument('--all', action='store_true',
                        help="Run all four combinations")
    args = parser.parse_args()

    pathlib.Path('outputs').mkdir(exist_ok=True)

    combos = [('loglinear','20news'),
              ('loglinear','hoc'),
              ('bert','20news'),
              ('bert','hoc')] if args.all else [(args.model, args.dataset)]

    for m, d in combos:
        experiment(m, d)

if __name__ == '__main__':
    main()