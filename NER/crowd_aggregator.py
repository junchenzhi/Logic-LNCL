import numpy as np




class CrowdsSequenceAggregator():

    def __init__(self, model, sample_weight, data_train, answers, num_classes, label2ind, batch_size=16, pi_prior=1.0, C=6.0):
        self.model = model
        self.sample_weight = sample_weight
        self.data_train = data_train
        self.answers = answers
        self.batch_size = batch_size
        self.pi_prior = pi_prior
        self.n_train = answers.shape[0]
        self.seq_length = answers.shape[1]
        self.num_classes = num_classes
        self.num_annotators = answers.shape[2]
        self.tag_to_ix = label2ind
        self.K = 0.1
        self.C = C

        print("n_train:", self.n_train)
        print("seq_length:", self.seq_length)
        print("num_classes:", self.num_classes)
        print("num_annotators:", self.num_annotators)

        # initialize annotators as reliable (almost perfect)
        self.pi = self.pi_prior * np.ones((self.num_classes, self.num_classes, self.num_annotators))

        # initialize estimated ground truth with majority voting
        self.ground_truth_est = np.zeros((self.n_train, self.seq_length, self.num_classes))
        for i in range(self.n_train):
            for j in range(self.seq_length):
                # votes = np.zeros(self.num_annotators)
                votes = np.zeros(self.num_classes)
                for r in range(self.num_annotators):
                    if answers[i, j, r] != -1:
                        votes[answers[i, j, r]] += 1
                self.ground_truth_est[i, j, np.argmax(votes)] = 1.0

        self.ground_truth_est_2 = np.zeros((self.n_train, self.seq_length, self.num_classes))
        self.ground_truth_est_3 = np.zeros((self.n_train, self.seq_length, self.num_classes))


        self.transitions = np.zeros((9, 10))
        '''
        In addition to the transition rules listed in Equations 16 and 17 in the paper, 
        we used similar transition rules whose non-zero weights (i.e., the non-zero elements in "self.transitions") are shown below.
        '''
        self.transitions[0] = np.array([0.873, 0.014, 0.018, 0.009, 0.0226, 0.018, 0.032, 0.009, 0.005, 0.000])
        self.transitions[:, -1] = np.array([0.55, 0.1, 0.0, 0.0, 0.3, 0.0, 0.05, 0.0, 0.0])
        self.transitions[1][0] = 1.0
        self.transitions[2][1] = 1.0
        self.transitions[3][0] = 1.0
        self.transitions[4][0] = 1.0
        self.transitions[5][4] = 0.8
        self.transitions[5][5] = 0.2
        self.transitions[6][0] = 1.0
        self.transitions[7][6] = 1.0
        self.transitions[8][3] = 0.5
        self.transitions[8][8] = 0.5
        self.transitions[self.transitions == 0] = 1e-10
        self.transitions = np.log(self.transitions)
        self.start = self.transitions[:, -1]
        self.transitions = self.transitions[:, :-1]



    def e_step(self):
        print("Pseudo-E-step...")
        self.ground_truth_est_3 = self.model.predict(self.data_train)
        # self.ground_truth_est = self.model.predict(self.data_train)
        self.ground_truth_est = self.ground_truth_est_3
        adjustment_factor = np.ones((self.n_train, self.seq_length, self.num_classes))
        for i in range(self.n_train):
            for j in range(self.seq_length):
                for r in range(self.num_annotators):
                    if self.answers[i, j, r] != -1:
                        adjustment_factor[i][j] *= self.pi[r, :, self.answers[i][j][r]]
        self.ground_truth_est = adjustment_factor * self.ground_truth_est
        self.ground_truth_est = self.ground_truth_est / np.sum(self.ground_truth_est, 2).reshape(
            (self.n_train, self.seq_length, 1))


    def e_step_model_predict(self):
        print("E-step...")
        return self.model.predict(self.data_train)


    def m_step(self, epochs):
        print("Pseudo-M-step...")
        hist = self.model.fit(self.data_train, self.ground_truth_est, epochs=1, shuffle=True,
                              batch_size=self.batch_size, verbose=0, sample_weight=self.sample_weight)
        # print("loss:", hist.history['loss'][-1])
        print("history.history.keys():", hist.history.keys())

        self.pi = np.zeros((self.num_annotators, self.num_classes, self.num_classes))
        for r in range(self.num_annotators):
            normalizer = np.zeros(self.num_classes)
            for i in range(self.n_train):
                for j in range(self.seq_length):
                    if self.answers[i, j, r] != -1:
                        normalizer += self.ground_truth_est[i, j, :]
                        self.pi[r, :, self.answers[i, j, r]] += self.ground_truth_est[i, j, :]
            normalizer[normalizer == 0] = 0.00001
            self.pi[r] = self.pi[r] / normalizer.reshape(self.num_classes, 1)
        return self.model, self.pi


    def decode(self, nonlogsoft, answers, flag=True):
        soft = nonlogsoft
        soft[soft == 0] = 1e-10
        soft = np.log(soft)
        tagset_size = soft.shape[2]
        sent_num = soft.shape[0]
        sent_len = soft.shape[1]
        hard_distribution = np.zeros(soft.shape)
        for sent in range(sent_num):
            s_len = len(answers[sent])
            backpointers = []

            forward_var = soft[sent][-s_len][1:].reshape(1, -1) + self.K * self.start
            for index, feat in enumerate(soft[sent]):
                if index <= sent_len - s_len:
                    continue
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step
                for next_tag in range(tagset_size - 1):
                    next_tag_var = forward_var + self.K * self.transitions[next_tag]
                    best_tag_id = np.argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id])
                forward_var = (np.array(viterbivars_t) + feat[1:]).reshape(1, -1)
                backpointers.append(bptrs_t)
            best_tag_id = np.argmax(forward_var)

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            best_path.reverse()
            for i in range(sent_len):
                if i < sent_len - s_len:
                    hard_distribution[sent][i][0] = 1
                else:
                    hard_distribution[sent][i][best_path[i - sent_len + s_len] + 1] = 1

        distr = np.exp(self.C * hard_distribution)
        nonlogsoft *= distr
        snt = nonlogsoft / np.sum(nonlogsoft, 2).reshape((sent_num, -1, 1))
        if flag:
            self.ground_truth_est_2 = snt

        return snt, hard_distribution


    def get_final_groud_truth(self, pai):
        res = (1-pai)*self.ground_truth_est + pai * self.ground_truth_est_2
        self.ground_truth_est = res


    def get_pai(self, cur_iter, params=None):
        """ exponential decay: pi_t = max{1 - k^t, lb} """
        k, lb = params[0], params[1]
        pi = 1. - max([k ** cur_iter, lb])
        return pi
