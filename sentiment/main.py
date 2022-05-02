from model import CNN
import utils_our
from logicnn_class import LogicNN
from fol import FOL_But

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import math
import os


def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin",
                                                         binary=True)
        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])
    logic_nn = LogicNN(network=model, C=params['C'])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion_2 = nn.KLDivLoss()

    all_train_acc = []
    all_test_acc = []
    all_dev_acc = []
    all_p_acc, all_snt_acc = [], []
    all_train_logic_acc = []
    all_test_logic_acc = []
    all_dev_logic_acc = []
    qft = []

    count = 0
    max_dev_acc = 0


    # initialize weighted_pred
    weighted_pred = begin_inference(data["answers"])

    flag = False
    for e in range(params["EPOCH"]):
        # m-step:worker_ability
        worker_ability = m_step_updates_worker_ability(truth=weighted_pred.cpu().numpy(), answers=data["answers"])
        weighted_pred = weighted_pred.cpu().numpy().tolist()
        data["train_x"], data["train_y"], data["train_but_fea"], data["train_but_ind"], data["answers"], weighted_pred = \
            shuffle(data["train_x"], data["train_y"], data["train_but_fea"], data["train_but_ind"], data["answers"],
                    weighted_pred)
        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)
            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_weighted_pred = weighted_pred[i:i + batch_range]

            batch_x = (torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_weighted_pred = torch.tensor(batch_weighted_pred).cuda(params["GPU"])

            # m-step:learn classifier
            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            m = nn.LogSoftmax(dim=1)
            softmax_pred = m(pred)
            loss = criterion_2(softmax_pred, batch_weighted_pred)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        train_x = [[data["word_to_idx"][w] for w in sent] + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] -
                                                                                          len(sent)) for sent in
                   data["train_x"]]
        train_but_fea = [[data["word_to_idx"][w] for w in sent] + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] -
                                                                                                len(sent)) for sent in
                         data["train_but_fea"]]
        train_x = (torch.LongTensor(train_x)).cuda(params["GPU"])
        train_but_fea = (torch.LongTensor(train_but_fea)).cuda(params["GPU"])
        f_but_ind = (torch.LongTensor(data["train_but_ind"])).cuda(params["GPU"])

        nclasses = params["CLASS_SIZE"]
        # e-step
        y_posterior = e_step(train_x, data["answers"], model, worker_ability, nclasses).cuda(params["GPU"]) 
        y_posterior_but = e_step_noestep(train_but_fea, data["answers"], model, worker_ability, nclasses).cuda(
            params["GPU"])  

        f_but_ind = f_but_ind.reshape((len(data["train_but_ind"]), 1))
        f_but_full = torch.cat((f_but_ind, y_posterior_but), 1)  

        rules = [FOL_But(nclasses, train_x, f_but_full)]
        logic_nn.input, logic_nn.rules = train_x, rules 
        snt_logic = logic_nn.cal_logic(y_posterior.clone())  

   
        new_pi = get_pi(cur_iter=(e + 1), params=params['pi_params'])
        all_p_acc.append(test_2(y_posterior, data))
        all_snt_acc.append(test_2(snt_logic, data))
        weighted_pred = ((1.0 - new_pi) * y_posterior + new_pi * snt_logic)
        qft.append(test_2(weighted_pred, data))
        print("Epoch:", e + 1)
        print("Inference, qft_acc:", qft[len(qft) - 1])

        if flag == False:
            train_acc_1 = test(data, model, params, mode="train")
            dev_acc_1 = test(data, model, params, mode="dev")
            test_acc_1 = test(data, model, params, mode="test")
            print("Prediction, Logic-LNCL-student: / train_acc:", train_acc_1, "/ dev_acc:", dev_acc_1, "/ test_acc:", test_acc_1)
            all_train_acc.append(train_acc_1)
            all_dev_acc.append(dev_acc_1)
            all_test_acc.append(test_acc_1)

            train_acc_2 = test_logic(data, model, logic_nn, params, pi=new_pi, mode="train")
            dev_acc_2 = test_logic(data, model, logic_nn, params, pi=new_pi, mode="dev")
            test_acc_2 = test_logic(data, model, logic_nn, params, pi=new_pi, mode="test")
            print("Prediction, Logic-LNCL-teacher: / train_acc:", train_acc_2, "/ dev_acc:", dev_acc_2, "/ test_acc:", test_acc_2)
            all_train_logic_acc.append(train_acc_2)
            all_dev_logic_acc.append(dev_acc_2)
            all_test_logic_acc.append(test_acc_2)

        if flag == False and dev_acc_1 > max_dev_acc:
            max_dev_acc = dev_acc_1
            max_test_acc_student = test_acc_1
            max_test_acc_teacher = test_acc_2
            count = 0
            best_model = copy.deepcopy(model)
        else:
            count += 1



        print("--------------------------------------------------")

        if params["EARLY_STOPPING"] and count == params["PATIENCE"]:
            print("early stopping by dev_acc!")
            np.save("worker_ability.npy", worker_ability)
            np.save(os.path.join(params["result_path"], 'worker_ability.npy'), worker_ability)
            # break
            flag = True
            '''
            When the prediction model is shown to have reached the early-stopping time on the validation set, 
            we need to continue to explore the inference performance. 
            '''



        if params["LEARNING_DECAY"]:
            lr = params['LEARNING_RATE'] * (0.5 ** (e // 5))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Results of the early-stopping time:
    print("max dev acc:", max_dev_acc, "test acc_student:", max_test_acc_student, "test acc_teacher:", max_test_acc_teacher)



    np.save(os.path.join(params["result_path"], 'all_p_acc.npy'), all_p_acc)
    np.save(os.path.join(params["result_path"], 'all_snt_acc.npy'), all_snt_acc)
    np.save(os.path.join(params["result_path"], 'qft.npy'), qft)

    np.save(os.path.join(params["result_path"], 'all_train_acc.npy'), all_train_acc)
    np.save(os.path.join(params["result_path"], 'all_dev_acc.npy'), all_dev_acc)
    np.save(os.path.join(params["result_path"], 'all_test_acc.npy'), all_test_acc)

    np.save(os.path.join(params["result_path"], 'all_train_logic_acc.npy'), all_train_logic_acc)
    np.save(os.path.join(params["result_path"], 'all_dev_logic_acc.npy'), all_dev_logic_acc)
    np.save(os.path.join(params["result_path"], 'all_test_logic_acc.npy'), all_test_logic_acc)

    return best_model


def test_logic(data, model, logicnn, params, pi, mode="test"):
    model.eval()
    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
        x_but_fea, x_but_ind = data["dev_but_fea"], data["dev_but_ind"]
    elif mode == "train":
        x, y = data["train_x"], data["train_y"]
        x_but_fea, x_but_ind = data["train_but_fea"], data["train_but_ind"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]
        x_but_fea, x_but_ind = data["test_but_fea"], data["test_but_ind"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = (torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]



    x_but_fea = [[data["word_to_idx"][w] for w in sent] + [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] -
                                                                                        len(sent)) for sent in
                 x_but_fea]
    x_but_fea = (torch.LongTensor(x_but_fea)).cuda(params["GPU"])
    x_but_ind = (torch.LongTensor(x_but_ind)).cuda(params["GPU"])

    nclasses = params["CLASS_SIZE"]
    y_posterior = F.softmax(model(x), dim=-1)  # y infer
    y_posterior_but = F.softmax(model(x_but_fea), dim=-1)  

    x_but_ind = x_but_ind.reshape((len(x_but_ind), 1))
    x_but_full = torch.cat((x_but_ind, y_posterior_but), 1)  


    rules = [FOL_But(nclasses, x, x_but_full)]
    logicnn.input, logicnn.rules = x, rules

    snt_logic = logicnn.cal_logic(y_posterior)  

    pred = np.argmax(snt_logic.cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def test_2(x, data):
    y = data["train_y"]
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(x.cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def test(data, model, params, mode="test"):
    model.eval()
    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "train":
        x, y = data["train_x"], data["train_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = (torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc





def get_pi(cur_iter, params=None, pi=None):
    """ exponential decay: pi_t = max{1 - k^t, lb} """
    k, lb = params[0], params[1]
    pi = 1. - max([k ** cur_iter, lb])
    return pi


def begin_inference(answers):
    sent_num = len(answers)
    worker_num = len(answers[0])
    y = np.zeros((sent_num, 2))
    for i in range(sent_num):
        num_0, num_1 = 0, 0
        for j in range(worker_num):
            if answers[i][j] == 0:
                num_0 += 1
            if answers[i][j] == 1:
                num_1 += 1
        y[i][0] = num_0 / (num_0 + num_1)
        y[i][1] = num_1 / (num_0 + num_1)
    return torch.tensor(y).cuda(0)


def m_step_updates_worker_ability(truth, answers):
    sent_num, worker_num = len(answers), len(answers[0])
    worker_ability = np.zeros((worker_num, 2, 2))
    worker_ability_before = np.zeros((worker_num, 2, 2))
    for j in range(worker_num):
        neg, pos = 0, 0
        for i in range(sent_num):
            if answers[i][j] != -1:
                neg += truth[i][0]
                pos += truth[i][1]
            if answers[i][j] == 0:
                worker_ability_before[j][0][0] += truth[i][0]
                worker_ability_before[j][1][0] += truth[i][1]
            worker_ability[j][0][0] = worker_ability_before[j][0][0] / neg if neg != 0 else 0.000001
            worker_ability[j][0][1] = 1 - worker_ability[j][0][0]
            worker_ability[j][1][0] = worker_ability_before[j][1][0] / pos if pos != 0 else 0.000001
            worker_ability[j][1][1] = 1 - worker_ability[j][1][0]

    return worker_ability


def e_step(x, answers, model, worker_ability, num_classes):
    model.eval()
    num_annotators = len(answers[0])
    pred = F.softmax(model(x), dim=-1).detach().cpu().numpy()

    adjustment_factor = np.ones((x.shape[0], num_classes))
    for i in range(x.shape[0]):
        for r in range(num_annotators):
            if answers[i][r] != -1:
                if answers[i][r] == 0:
                    adjustment_factor[i] *= worker_ability[r, :, 0]
                else:
                    adjustment_factor[i] *= worker_ability[r, :, 1]
    pred = adjustment_factor * pred
    pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
    return torch.tensor(pred)



def e_step_noestep(x, answers, model, worker_ability, num_classes):
    model.eval()
    num_annotators = len(answers[0])
    pred = F.softmax(model(x), dim=-1).detach().cpu().numpy()

    adjustment_factor = np.ones((x.shape[0], num_classes))
    for i in range(x.shape[0]):
        for r in range(num_annotators):
            if answers[i][r] != -1:
                if answers[i][r] == 0:
                    adjustment_factor[i] *= worker_ability[r, :, 0]
                else:
                    adjustment_factor[i] *= worker_ability[r, :, 1]
    return torch.tensor(pred)


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="static", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="MR", help="available datasets: MR")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stop", default=True, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--learn_decay", default=True, action='store_true', help="whether to apply learning decay")
    parser.add_argument("--epoch", default=30, type=int, help="number of max epoch")
    parser.add_argument("--lr", default=1, type=float, help="learning rate")
    parser.add_argument("--dr", default=0.5, type=float, help="DROPOUT_PROB")
    parser.add_argument("--C", default=5.0, type=float, help="regularization strength")
    parser.add_argument("--p1", default=0.94, type=float, help="p1")
    parser.add_argument("--p2", default=0., type=float, help="p2")
    parser.add_argument("--patience", default=5, type=int, help="early stopping patience")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument('--result_path')

    options = parser.parse_args()

    data = getattr(utils_our, f"read_{options.dataset}")()
    data["vocab"] = (list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = [0, 1]
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stop,
        "LEARNING_DECAY": options.learn_decay,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.lr,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": options.dr,
        "NORM_LIMIT": 3,
        "pi_params": [options.p1, options.p2],
        "C": options.C,
        "PATIENCE": options.patience,
        "GPU": options.gpu,
        "result_path": "./results/" + options.result_path
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("TRAIN", len(data["train_x"]))
    print("DEV", len(data["dev_x"]))
    print("TEST", len(data["test_x"]))
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("LEARNING_DECAY:", params["LEARNING_DECAY"])
    print("PATIENCE:", params["PATIENCE"])
    print("PI:", params["pi_params"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("MAX_SENT_LEN:", params["MAX_SENT_LEN"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if not os.path.exists(params["result_path"]):
        os.makedirs(params["result_path"])

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils_our.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils_our.load_model(params).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
