from sklearn.utils import shuffle
import re
import pickle
from collections import defaultdict
import chardet
import numpy as np
from collections import defaultdict
import sys, re
import pandas as pd


def read_MR():
    data = {}
    x_train, y_train, train_but_fea, train_but_ind = [], [], [], []
    x_test, y_test, test_but_fea, test_but_ind = [], [], [], []
    but_fea_cnt = 0
    train_file = "./data/mturk_answers.csv"
    test_file = "./data/test_answer.csv"

    df = pd.read_csv(train_file)
    workers_id_map = df['WorkerId'].unique()
    sent_id_map = df['Input.id'].unique()
    len_sent = len(df['Input.id'].unique())
    len_workers = len(df['WorkerId'].unique())
    used = np.zeros(len_sent)
    answer = np.ones((len_sent, len_workers), dtype=int) * -1
    for row in df.iterrows():
        line = (row[1]['Input.original_sentence'])
        # line = line.decode(encoding='utf-8',errors='ignore')
        worker = np.where(workers_id_map == row[1]['WorkerId'])[0][0]
        sent = np.where(sent_id_map == row[1]['Input.id'])[0][0]
        worker_label = 1 if row[1]['Answer.sent'] == 'pos' else 0
        answer[sent][worker] = worker_label
        if used[sent] > 0:
            continue
        used[sent] = 1
        rev = []
        rev.append(line.strip())
        orig_line = clean_str(" ".join(rev))
        x_train.append(orig_line.split())
        y = 1 if row[1]['Input.true_sent'] == 'pos' else 0
        y_train.append(y)
        if ' but ' in orig_line:
            train_but_ind.append(1)
            fea = orig_line.split(' but ')[1:]
            fea = ' '.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            train_but_ind.append(0)
            fea = ''
        train_but_fea.append(fea.split())

    df = pd.read_csv(test_file)
    for row in df.iterrows():
        line = (row[1]['Input.original_sentence'])
        rev = []
        rev.append(line.strip())
        orig_line = clean_str(" ".join(rev))
        x_test.append(orig_line.split())
        y_test.append(int(row[1]['Input.true_sent']))
        if ' but ' in orig_line:
            test_but_ind.append(1)
            fea = orig_line.split(' but ')[1:]
            fea = ' '.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            test_but_ind.append(0)
            fea = ''
        test_but_fea.append(fea.split())
    print("but count ", but_fea_cnt)

    x_train, y_train, train_but_fea, train_but_ind, answer = shuffle(x_train, y_train, train_but_fea, train_but_ind, answer)
    x_test, y_test, test_but_fea, test_but_ind = shuffle(x_test, y_test, test_but_fea, test_but_ind)
    dev_idx = 3000

    # train
    data["train_x"], data["answers"], data["train_y"] = x_train, answer.tolist(), y_train
    data["train_but_fea"], data["train_but_ind"] = train_but_fea, train_but_ind
    # val
    data["dev_x"], data["dev_y"] = x_test[:dev_idx], y_test[:dev_idx]
    data["dev_but_fea"], data["dev_but_ind"] = test_but_fea[:dev_idx], test_but_ind[:dev_idx]
    # test
    data["test_x"], data["test_y"] = x_test[dev_idx:], y_test[dev_idx:]
    data["test_but_fea"], data["test_but_ind"] = test_but_fea[dev_idx:], test_but_ind[dev_idx:]
    return data


def read_MR_baseline():
    data = {}
    x_train, y_train, train_but_fea, train_but_ind, glad, mv = [], [], [], [], [], []
    x_test, y_test, test_but_fea, test_but_ind = [], [], [], []
    but_fea_cnt = 0
    train_file = "./data/glad.csv"
    test_file = "./data/test_answer.csv"

    df = pd.read_csv(train_file)
    workers_id_map = df['WorkerId'].unique()
    sent_id_map = df['Input.id'].unique()
    len_sent = len(df['Input.id'].unique())
    len_workers = len(df['WorkerId'].unique())
    used = np.zeros(len_sent)
    answer = np.ones((len_sent, len_workers), dtype=int) * -1
    for row in df.iterrows():
        line = (row[1]['Input.original_sentence'])
        # line = line.decode(encoding='utf-8',errors='ignore')
        worker = np.where(workers_id_map == row[1]['WorkerId'])[0][0]
        sent = np.where(sent_id_map == row[1]['Input.id'])[0][0]
        worker_label = 1 if row[1]['Answer.sent'] == 'pos' else 0
        answer[sent][worker] = worker_label
        if used[sent] > 0:
            continue
        used[sent] = 1
        rev = []
        rev.append(line.strip())
        orig_line = clean_str(" ".join(rev))
        x_train.append(orig_line.split())
        y = 1 if row[1]['Input.true_sent'] == 'pos' else 0
        y_train.append(y)
        glad.append(int(row[1]["glad"]))
        if ' but ' in orig_line:
            train_but_ind.append(1)
            fea = orig_line.split(' but ')[1:]
            fea = ' '.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            train_but_ind.append(0)
            fea = ''
        train_but_fea.append(fea.split())

    df = pd.read_csv(test_file)
    for row in df.iterrows():
        line = (row[1]['Input.original_sentence'])
        rev = []
        rev.append(line.strip())
        orig_line = clean_str(" ".join(rev))
        x_test.append(orig_line.split())
        y_test.append(int(row[1]['Input.true_sent']))
        if ' but ' in orig_line:
            test_but_ind.append(1)
            fea = orig_line.split(' but ')[1:]
            fea = ' '.join(fea)
            fea = fea.strip().replace('  ', ' ')
            but_fea_cnt += 1
        else:
            test_but_ind.append(0)
            fea = ''
        test_but_fea.append(fea.split())
    print("but count ", but_fea_cnt)

    x_train, y_train, train_but_fea, train_but_ind, answer, glad = shuffle(x_train, y_train, train_but_fea, train_but_ind, answer, glad)
    x_test, y_test, test_but_fea, test_but_ind = shuffle(x_test, y_test, test_but_fea, test_but_ind)
    # dev_idx = len(x_test) // 10 * 5
    dev_idx = 3000

    # train
    data["train_x"], data["answers"], data["train_y"], data["glad"] = x_train, answer.tolist(), y_train, glad
    data["train_but_fea"], data["train_but_ind"] = train_but_fea, train_but_ind
    # val
    data["dev_x"], data["dev_y"] = x_test[:dev_idx], y_test[:dev_idx]
    data["dev_but_fea"], data["dev_but_ind"] = test_but_fea[:dev_idx], test_but_ind[:dev_idx]
    # test
    data["test_x"], data["test_y"] = x_test[dev_idx:], y_test[dev_idx:]
    data["test_but_fea"], data["test_but_ind"] = test_but_fea[dev_idx:], test_but_ind[dev_idx:]
    return read_MR_mv(data)


def clean_str(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_MR_mv(data):
    answers = np.array(data["answers"])
    n_train, num_annotators = answers.shape[0], answers.shape[1]
    # initialize estimated ground truth with majority voting
    y_mv = [0] * n_train
    for i in range(n_train):
        votes = np.zeros(2)
        for r in range(num_annotators):
            if answers[i, r] != -1:
                votes[answers[i, r]] += 1
        y_mv[i] = votes.argmax()
    data["mv"] = y_mv

    return data


def make_glad_file():
    data = read_MR()
    origin_file = "./data/origin_glad.csv"
    train_file = "./data/mturk_answers.csv"
    glad_file = "./data/glad.csv"
    df_1 = pd.read_csv(origin_file)
    df = pd.read_csv(train_file)
    glad = []
    for row in df.iterrows():
        id = row[1]['Input.id']
        g_row = df_1.loc[df_1["question"] == id].iat[0, 1]
        glad.append(g_row)
    df["glad"] = glad
    df.to_csv(glad_file, index=False)
    return data


def save_model(model, params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()


data = read_MR()