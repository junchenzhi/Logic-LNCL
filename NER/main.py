import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv1D
from keras.layers.recurrent import GRU
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score
from conlleval import conlleval
from crowd_aggregator import CrowdsSequenceAggregator
import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--result_path')
parser.add_argument('--fewer_sample', default=None, type=str, help="fewer training data")
parser.add_argument("--validation_teacher", default=False,
                    help="observe the teacher's performance on the validation data")
options = parser.parse_args()
params = {
    "result_path": options.result_path,
    "fewer_sample": options.fewer_sample,
    "validation_teacher": options.validation_teacher
}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

NUM_RUNS = 30
DATA_PATH = "./data/deep_ner-mturk/ner-mturk/"
EMBEDDING_DIM = 300
BATCH_SIZE = 64

embeddings_index = {}
f = open("./data/glove.6B.%dd.txt" % (EMBEDDING_DIM,))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors' % len(embeddings_index))


def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x


all_answers = read_conll(DATA_PATH+'answers.txt')
all_mv = read_conll(DATA_PATH+'mv.txt')
all_ground_truth = read_conll(DATA_PATH+'ground_truth.txt')
all_test = read_conll(DATA_PATH+'testset.txt')
all_docs = all_ground_truth + all_test
print("Answers data size:", len(all_answers))
print("Majority voting data size:", len(all_mv))
print("Ground truth data size:", len(all_ground_truth))
print("Test data size:", len(all_test))
print("Total sequences:", len(all_docs))


X_train = [[c[0] for c in x] for x in all_answers]
y_answers = [[c[1:] for c in y] for y in all_answers]
y_mv = [[c[1] for c in y] for y in all_mv]
y_ground_truth = [[c[1] for c in y] for y in all_ground_truth]
X_test = [[c[0] for c in x] for x in all_test]
y_test = [[c[1] for c in y] for y in all_test]
X_all = [[c[0] for c in x] for x in all_docs]
y_all = [[c[1] for c in y] for y in all_docs]
N_ANNOT = len(y_answers[0][0])
print("Num annnotators:", N_ANNOT)

lengths = [len(x) for x in all_docs]
all_text = [c for x in X_all for c in x]
words = list(set(all_text))
word2ind = {word: index for index, word in enumerate(words)}
ind2word = {index: word for index, word in enumerate(words)}
labels = list(set([c for x in y_all for c in x]))
print("Labels:", labels)
label2ind = {"pad":0,"O":1,"B-PER":2, "I-PER":3,"B-MISC":4, "B-ORG":5, "I-ORG":6, "B-LOC":7, "I-LOC":8, "I-MISC":9}
ind2label = {label2ind[label]: label for label in (label2ind)}
print('Input sequence length range: ', max(lengths), min(lengths))
max_label = max(label2ind.values()) + 1
print("Max label:", max_label)
maxlen = max([len(x) for x in X_all])
print('Maximum sequence length:', maxlen)




# # Prepare embedding matrix
num_words = len(word2ind)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2ind.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result


X_train_enc = [[word2ind[c] for c in x] for x in X_train]
y_ground_truth_enc = [([0] * (maxlen - 1 - len(ey))) + [label2ind[c] for c in ey] for ey in y_ground_truth]
y_ground_truth_enc = [[encode(c, max_label) for c in ey] for ey in y_ground_truth_enc]
y_mv_enc = [[0] * (maxlen - 1 - len(ey)) + [label2ind[c] for c in ey] for ey in y_mv]
y_mv_enc = [[encode(c, max_label) for c in ey] for ey in y_mv_enc]

y_answers_enc = []
for r in range(N_ANNOT):
    annot_answers = []
    for i in range(len(y_answers)):
        padnum = maxlen - len(y_answers[i])
        seq = [0] * padnum
        for j in range(len(y_answers[i])):
            enc = -1
            if y_answers[i][j][r] != "?":
                enc = label2ind[y_answers[i][j][r]]
            seq.append(enc)
        # seq.append(11)
        annot_answers.append(seq)
    y_answers_enc.append(annot_answers)



final_my_need = []
for i in range(len(y_answers)):
    this_num = 0
    for r in range(N_ANNOT):
        if y_answers[i][0][r] != "?":
            this_num += 1
    final_my_need.append(this_num)
mean = sum(final_my_need)/len(final_my_need)
final_my_need = [x/mean for x in final_my_need]
final_my_need = np.array(final_my_need)




X_test, y_test = shuffle(X_test, y_test)
X_test_enc = [[word2ind[c] for c in x] for x in X_test]
y_test_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y_test]
y_test_enc = [[encode(c, max_label) for c in ey] for ey in y_test_enc]

# pad sequences
X_train_enc = pad_sequences(X_train_enc, maxlen=maxlen)
y_ground_truth_enc = pad_sequences(y_ground_truth_enc, maxlen=maxlen)
y_mv_enc = pad_sequences(y_mv_enc, maxlen=maxlen)
X_test_enc = pad_sequences(X_test_enc, maxlen=maxlen)
y_test_enc = pad_sequences(y_test_enc, maxlen=maxlen)

dev_idx = 2000
X_test_enc, X_val_enc = X_test_enc[:dev_idx], X_test_enc[dev_idx:]
y_test_enc, y_val_enc = y_test_enc[:dev_idx], y_test_enc[dev_idx:]

print("test count: ", X_test_enc.shape[0])
print("val count: ", X_val_enc.shape[0])
y_answers_enc_padded = []
for r in range(N_ANNOT):
    padded_answers = pad_sequences(y_answers_enc[r], maxlen=maxlen)
    y_answers_enc_padded.append(padded_answers)

y_answers_enc_padded = np.array(y_answers_enc_padded)
y_answers_enc = np.transpose(y_answers_enc_padded, [1, 2, 0])

n_train = len(X_train_enc)
n_test = len(X_test_enc)

print('Training and testing tensor shapes:')
print(X_train_enc.shape, X_test_enc.shape, y_ground_truth_enc.shape, y_test_enc.shape)

print("Answers shape:", y_answers_enc.shape)
N_CLASSES = len(label2ind)
print("Num classes:", N_CLASSES)


if params['fewer_sample'] != None:
    print("\n")
    print("=" * 20 + "Experiments on fewer samples" + "=" * 20)
    print('Now we use fewer samples for training, i.e, ', params['fewer_sample'], 'samples')
    print("=" * 20 + "Experiments on fewer samples" + "=" * 20)
    final_my_need, X_train_enc, y_answers_enc, y_ground_truth_enc, y_answers, X_train, y_ground_truth = shuffle(final_my_need, X_train_enc, y_answers_enc, y_ground_truth_enc, y_answers, X_train, y_ground_truth)
    final_my_need = final_my_need[: int(params['fewer_sample'])]
    X_train_enc = X_train_enc[: int(params['fewer_sample'])]
    y_answers_enc = y_answers_enc[: int(params['fewer_sample'])]
    y_ground_truth_enc = y_ground_truth_enc[: int(params['fewer_sample'])]
    y_answers = y_answers[: int(params['fewer_sample'])]
    X_train = X_train[: int(params['fewer_sample'])]
    y_ground_truth = y_ground_truth[: int(params['fewer_sample'])]


# Here we shall use features representation produced by the VGG16 network as the input. Our base model is then simply composed by one densely-connected layer with 128 hidden units and an output dense layer. We use 50% dropout between the two dense layers.
def build_base_model():
    base_model = Sequential()
    base_model.add(Embedding(num_words,
                        300,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=True))
    base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
    base_model.add(Dropout(0.5))
    base_model.add(GRU(50, return_sequences=True))
    base_model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))
    base_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return base_model


# # Auxiliary functions for evaluating the models
def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr


def eval_tiaoshi(pr_test, y_truth):
    pr_test = np.argmax(pr_test, axis=2)
    yh = y_truth.argmax(2)
    fyh, fpr = score(yh, pr_test)
    acc = accuracy_score(fyh, fpr)

    preds_test = []
    for i in range(len(pr_test)):
        coords = -len(y_ground_truth[i])
        row = pr_test[i][coords:]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    preds_test = [list(map(lambda x: ind2label[x], y)) for y in preds_test]

    results_test = conlleval(preds_test, y_ground_truth, X_train, "./r_test/" + params["result_path"] + ".txt")
    print("qft, accuracy:", acc, results_test)

    return results_test, acc


def eval_model(model, X_test_enc, y_test_enc, mode="test"):
    pr_test = model.predict(X_test_enc, verbose=2)
    pr_test = np.argmax(pr_test, axis=2)

    yh = y_test_enc.argmax(2)
    fyh, fpr = score(yh, pr_test)
    acc = accuracy_score(fyh, fpr)

    preds_test = []
    for i in range(len(pr_test)):
        coords = -len(y_test[i])
        if mode == "val":
            coords = -len(y_test[dev_idx + i])
        row = pr_test[i][coords:]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    preds_test = [list(map(lambda x: ind2label[x], y)) for y in preds_test]
    if mode == "test":
        x_, y_ = X_test[:dev_idx], y_test[:dev_idx]
    else:
        x_, y_ = X_test[dev_idx:], y_test[dev_idx:]
    results_test = conlleval(preds_test, y_, x_, "./r_test/" + params["result_path"] + ".txt")


    if mode == "test":
        print("Logic-LNCL-student on test data, accuracy:", acc, results_test)
    else:
        print("Logic-LNCL-student on validation data, accuracy:", acc, results_test)

    return results_test, acc



def eval_model_logic(model, X_test_enc, y_test_enc, mode="test"):
    if mode == "test":
        x_, y_ = X_test[:dev_idx], y_test[:dev_idx]
    else:
        x_, y_ = X_test[dev_idx:], y_test[dev_idx:]

    pr_test = model.predict(X_test_enc, verbose=2)
    pr_test, _ = crowds_agg.decode(pr_test, y_, flag=False)
    pr_test = np.argmax(pr_test, axis=2)
    yh = y_test_enc.argmax(2)
    fyh, fpr = score(yh, pr_test)
    acc = accuracy_score(fyh, fpr)

    preds_test = []
    for i in range(len(pr_test)):
        coords = -len(y_test[i])
        if mode == "val":
            coords = -len(y_test[dev_idx + i])
        row = pr_test[i][coords:]
        row[np.where(row == 0)] = 1
        preds_test.append(row)
    preds_test = [list(map(lambda x: ind2label[x], y)) for y in preds_test]
    results_test = conlleval(preds_test, y_, x_, "./r_test/" + params["result_path"] + ".txt")

    if mode == "test":
        print("Logic-LNCL-teacher on test data, accuracy:", acc, results_test)
    else:
        print("Logic-LNCL-teacher on validation data, accuracy:", acc, results_test)

    return results_test, acc



model = build_base_model()
crowds_agg = CrowdsSequenceAggregator(model, final_my_need, X_train_enc, y_answers_enc, num_classes=N_CLASSES, label2ind=label2ind,
                                      batch_size=BATCH_SIZE, C=5.0)

'''
Here k_1 and k_2, the two hyper-parameters, correspond to ``imitation strength $k^{(t)}$'' in our paper.
Set k_2 to 0.20, not 0.10 as stated in our paper (we wrote this value wrong in our paper).
'''
# k(t) = min(0.80, 1-0.90**t)
k_1, k_2 = 0.90, 0.20

all_p, all_f1, all_r = [], [], []
all_logic_p, all_logic_f1, all_logic_r = [], [], []
all_val_f1, all_val_f1_logic = [], []
all_test_acc, all_logic_test_acc = [], []

hard_f1, p_f1 = [], []
qlt, qft = [], []
qft_precision, qft_recall =[], []

best_val_f1 = 0
ret_test_f1 = 0
ret_logic_test_f1 = 0
stop_count = 0
patience = 5
flag = False



results_dir = os.path.join("results", params["result_path"])
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
results_dir_r_test = "./r_test/"
if not os.path.exists(results_dir_r_test):
    os.makedirs(results_dir_r_test)

print('\n')
print('Begin training...')
for epoch in range(30):
    print('\n')
    print('Epoch:', epoch + 1)
    # Pseudo-M-step
    model, pi = crowds_agg.m_step(epoch)

    if flag == False:
        results_val, val_acc = eval_model(model, X_val_enc, y_val_enc, mode="val")
        results_val_logic, val_acc_logic = eval_model_logic(model, X_val_enc, y_val_enc, mode="val")
        results_test, test_acc = eval_model(model, X_test_enc, y_test_enc, mode="test")
        results_test_logic, test_logic_acc = eval_model_logic(model, X_test_enc, y_test_enc, mode="test")

        val_f1 = results_val["f1"]
        all_val_f1.append(results_val["f1"])

        all_p.append(results_test["p"])
        all_r.append(results_test["r"])
        all_f1.append(results_test["f1"])

        all_logic_p.append(results_test_logic["p"])
        all_logic_r.append(results_test_logic["r"])
        all_logic_f1.append(results_test_logic["f1"])

        all_val_f1_logic.append(results_val_logic["f1"])


    if params["validation_teacher"] == False:
        val_performance = results_val["f1"]
    else:
        val_performance = results_val_logic["f1"]

    if flag == False and val_performance > best_val_f1:
        best_val_f1 = val_performance

        student_f1 = results_test["f1"]
        student_precision = results_test["p"]
        student_recall = results_test["r"]

        teacher_f1 = results_test_logic["f1"]
        teacher_precision = results_test_logic["p"]
        teacher_recall = results_test_logic["r"]

        stop_count = 0
    else:
        stop_count += 1
    if stop_count == patience:
        np.save(os.path.join(results_dir, 'pi_earlystop.npy'), pi)
        flag = True
        # break


    # Pseudo-E-step
    crowds_agg.e_step()
    ground_truth_est_3 = crowds_agg.ground_truth_est_3
    # ground_truth_est is qbt
    ground_truth_est = crowds_agg.ground_truth_est
    ground_truth_est_logic, hard = crowds_agg.decode(ground_truth_est_3, y_answers)
    # pai = crowds_agg.get_pai(epoch+1, params=[k_1, k_2])
    pai = crowds_agg.get_pai(epoch + 1, [k_1, k_2])
    crowds_agg.get_final_groud_truth(pai)
    result_qft, _ = eval_tiaoshi(crowds_agg.ground_truth_est, y_ground_truth_enc)
    qft.append(result_qft['f1'])
    qft_precision.append(result_qft['p'])
    qft_recall.append(result_qft['r'])

qft = np.array(qft)
stop_time_inference = qft.argmax()
qft_f1 = max(qft)
qft_precision = qft_precision[stop_time_inference]
qft_recall = qft_recall[stop_time_inference]



print('Final results:')
print('Prediction:')
print('Logic-LNCL-student: F1:', student_f1, 'precision:', student_precision, 'recall:', student_recall)
print('Logic-LNCL-teacher: F1:', teacher_f1, 'precision:', teacher_precision, 'recall:', teacher_recall)

print('Inference:')
print('Inference, F1:', qft_f1, 'precision:', qft_precision,
      'recall:', qft_recall)



# validation set
np.save(os.path.join(results_dir, 'all_val_f1.npy'), all_val_f1)
np.save(os.path.join(results_dir, 'all_val_f1_logic.npy'), all_val_f1_logic)

# training set
np.save(os.path.join(results_dir, 'qft.npy'), qft)
np.save(os.path.join(results_dir, 'qft_precision.npy'), qft_precision)
np.save(os.path.join(results_dir, 'qft_recall.npy'), qft_recall)

# test data
np.save(os.path.join(results_dir, 'all_p.npy'), all_p)
np.save(os.path.join(results_dir, 'all_r.npy'), all_r)
np.save(os.path.join(results_dir, 'all_f1.npy'), all_f1)

np.save(os.path.join(results_dir, 'all_logic_f1.npy'), all_logic_f1)
np.save(os.path.join(results_dir, 'all_logic_p.npy'), all_logic_p)
np.save(os.path.join(results_dir, 'all_logic_r.npy'), all_logic_r)
