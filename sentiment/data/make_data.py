import csv
import os
import pandas as pd
import numpy as np


all_answer = "./all_answer.csv"
train_file = "./mturk_answers.csv"
test_file = "./test_answer.csv"
pos_file = "./rt-polarity.pos"
neg_file = "./rt-polarity.neg"


def make_all_answer():
    sents = []
    with open(pos_file, "rb") as f:
        for line in f:
            line = line.decode(encoding='utf-8', errors='ignore')
            line = line.strip()
            sents.append([line, 1])
    f.close()
    with open(neg_file, "rb") as f:
        for line in f:
            line = line.decode(encoding='utf-8', errors='ignore')
            line = line.strip()
            sents.append([line, 0])
    f.close()
    return sents


def remove_duplicate(sents):
    df = list(pd.read_csv(train_file)['Input.original_sentence'].unique())
    for i in range(len(df)):
        df[i] = df[i].strip()
    sents_copy = []
    for sent in sents:
        sen = sent[0]
        if sen not in df:
            sents_copy.append(sent)
    return sents_copy


def test(sents):
    count = 0
    df = list(pd.read_csv(train_file)['Input.original_sentence'].unique())
    sent = [c[0] for c in sents]
    for i in range(len(df)):
        df[i] = df[i].strip()
        if df[i] not in sent:
            print(df[i])
            print("------------")
            count += 1
    print(count)


if __name__ == "__main__":
    sents = make_all_answer()
    print("all sentence length is ", len(sents))
    df = pd.DataFrame(sents,columns=['Input.original_sentence', 'Input.true_sent'])
    df.to_csv(all_answer,index=False)

    new_sents = remove_duplicate(sents)
    print("test sentence length is ", len(new_sents))
    df = pd.DataFrame(new_sents,columns=['Input.original_sentence', 'Input.true_sent'])
    df.to_csv(test_file,index=False)
