from collections import defaultdict
from collections import Counter

import math


class HiddenMarkovModel:
    """Hidden markov model for part of speech tagging on the Penn Treebank dataset
    """

    def __init__(
        self, train="./pos_train.txt", test="./pos_test.txt", supervised=True):
        """
        Args:
            train: str. The path to the file containing the training data.
            test: str. The path to the file containing the testing data.
            supervised: bool. Whether or not to use the tags in the part of speech
                data.
        """
        self.epsilon = 0.000001
        self.testpath=test
        self.trainpath=train

        tag_counts = Counter()
        self.tag_given_tag_counts=dict()
        self.word_given_tag_counts=dict()

        with open (train ,"r") as infile:
            for line in infile:
                #
                # first tag is the start symbol
                lasttag="<s>"
                #
                # split line into word/tag pairs
                #
                for wordtag in line.rstrip().split(" "):
                    if wordtag == "":
                        continue
                    # note that you might have escaped slashes
                    # 1\/2/CD means "1/2" "CD"
                    # keep 1/2 as 1\/2 
                    parts=wordtag.split("/")
                    tag=parts.pop()
                    word="/".join(parts)
                    #
                    # update counters
                    if tag not in tag_counts:
                        tag_counts[tag] = 1
                    else:
                        tag_counts[tag] += 1

                    if tag not in self.word_given_tag_counts:
                        self.word_given_tag_counts[tag]=Counter()
                    if lasttag not in self.tag_given_tag_counts:
                        self.tag_given_tag_counts[lasttag]=Counter()
                    if supervised:
                        self.word_given_tag_counts[tag][word]+=1
                        self.tag_given_tag_counts[lasttag][tag]+=1
                    else:
                        self.word_given_tag_counts[tag][word]=1
                        self.tag_given_tag_counts[lasttag][tag]=1

                    lasttag=tag
                if lasttag not in self.tag_given_tag_counts:
                    self.tag_given_tag_counts[lasttag] = Counter()
                self.tag_given_tag_counts[lasttag]["</s>"]+=1

        self.mode_tag = tag_counts.most_common(1)[0][0]

        # Compute the probability matrices A and B
        self.trans_prob = defaultdict(lambda: defaultdict(float))
        for tag1 in self.tag_given_tag_counts.keys():
            norm = sum(self.tag_given_tag_counts[tag1].values())
            for tag2, count in self.tag_given_tag_counts[tag1].items():
                self.trans_prob[tag1][tag2] = count / norm


        self.vocabulary = set()
        self.obs_prob = defaultdict(lambda: defaultdict(float))
        smooth_adj = self.epsilon * len(self.word_given_tag_counts[self.mode_tag])

        for tag in self.word_given_tag_counts.keys():
            norm = sum(self.word_given_tag_counts[tag].values())
            if tag == self.mode_tag:
                adj = smooth_adj
                self.obs_prob[self.mode_tag]['<UNK>'] = self.epsilon / (norm + adj)
            else:
                adj = 0

            for word, count in self.word_given_tag_counts[tag].items():
                self.vocabulary.add(word)
                self.obs_prob[tag][word] = (count + self.epsilon) / (norm + adj)


        self.tags = self.tag_given_tag_counts.keys()

    def _forward(self, observations):
        """Forward step of training the HMM.

        Args:
            observations: A list of strings.

        Returns:
            A list of dict representing the trellis of alpha values
        """
        states = self.tag_given_tag_counts.keys()
        trellis = [{}] # Trellis to fill with alpha values
        for state in states:
            trellis[0][state] = (self.trans_prob["<s>"][state]
                * self.obs_prob[state][observations[0]])

        for t in range(1, len(observations)):
            trellis.append({})
            for state in states:
                trellis[t][state] = sum(
                    trellis[t-1][prev_state] * self.trans_prob[prev_state][state]
                    * self.obs_prob[state][observations[t]] for prev_state in states)

        # Terminal step
        trellis.append({})
        q_f = "</s>"
        trellis[t+1][q_f] = sum(
            trellis[t][prev_state]*self.trans_prob[prev_state][q_f]
            for prev_state in states)
        return trellis

    def _backward(self, observations):
        """Backward step of training the HMM.

        Args:
            observations: A list of strings.

        Returns:
            A list of dict representing the trellis of beta values
        """
        states = self.tag_given_tag_counts.keys()
        trellis = [{}]

        for state in states:
            trellis[0][state] = self.trans_prob[state]["</s>"]

        for t in range(len(observations)-1, 0, -1):
            trellis.insert(0, {})
            for state in states:
                trellis[0][state] = sum(trellis[1][next_state]
                    * self.trans_prob[state][next_state]
                    * self.obs_prob[next_state][observations[t]]
                    for next_state in states)

        trellis.insert(0, {})
        q_0 = "<s>"
        trellis[0][q_0] = sum(trellis[1][next_state] * self.trans_prob[q_0][next_state]
            * self.obs_prob[next_state][observations[t-1]] for next_state in states)
        return trellis

    def _compute_new_params(self, alphas, betas, observations):
        """Compute new transition and emission probabilities using the
        alpha and beta values. Should be used with supervised=False during
        object initialization.

        Args:
            alpha:
                list of dicts representing the alpha values. alpha[t]['state']
                is the forward probability for the state at timestep t.
            beta:
                list of dicts representing the beta values. beta[t]['state']
                is the backward probability for the state at timestep t
        """

        # E-step
        chi = []
        gamma = []
        for t in range(len(alphas)-2):
            chi.append(defaultdict(lambda: defaultdict(float))) 
            gamma.append(defaultdict(float))
            for state in self.tag_given_tag_counts.keys():
                for next_state in self.tag_given_tag_counts.keys():
                    chi[t][state][next_state] = (alphas[t][state] * self.trans_prob[state][next_state]
                        * self.obs_prob[next_state][observations[t+1]] * betas[t+2][next_state]) / alphas[-1]['</s>']
                gamma[t][state] = alphas[t][state] * betas[t+1][state] / alphas[-1]['</s>'] 

        # M-step
        for i in self.tag_given_tag_counts.keys():
            for j in self.tag_given_tag_counts.keys():
                self.trans_prob[i][j] = (sum(chi[t][i][j] for t in range(len(alphas)-2))
                / sum(chi[t][i][k] for t in range(len(alphas)-2) for k in self.tag_given_tag_counts.keys()))

            for v_k in observations:
                self.obs_prob[i][v_k] = sum(gamma[t][i] for t in len(observations) if observations[t] == v_k)
                self.obs_prob[i][v_k] /= sum(gamma[t][i] for t in len(observations))

    def viterbi(self, words):
        trellis = {}
        for tag in self.tags:
            trellis[tag] = [self.get_log_prob(self.trans_prob, '<s>', tag), ['<s>', tag]]
            if words[0] in self.vocabulary:
                trellis[tag][0] += self.get_log_prob(self.obs_prob, tag, words[0])
            else:
                trellis[tag] += self.get_log_prob(self.obs_prob, tag, '<UNK>')

        new_trellis = {}
        for word in words[1:]:
            for cur_tag in self.tags:
                cur_min_prob = float('inf')
                cur_min_path = None

                for prev_tag in self.tags:
                    prob = trellis[prev_tag][0] + self.get_log_prob(self.trans_prob, prev_tag, cur_tag)
                    if word in self.vocabulary:
                        prob += self.get_log_prob(self.obs_prob, cur_tag, word)
                    else:
                        prob += self.get_log_prob(self.obs_prob, cur_tag, '<UNK>')

                    if prob < cur_min_prob:
                        cur_min_prob = prob
                        cur_min_path = trellis[prev_tag][1] + [cur_tag]

                new_trellis[cur_tag] = [cur_min_prob, cur_min_path]

            trellis = new_trellis
            new_trellis = {}

        cur_min_prob = float('inf')
        cur_min_path = None
        for tag in self.tags:
            prob = self.get_log_prob(self.trans_prob, tag, '</s>') + trellis[tag][0]
            if prob < cur_min_prob:
                cur_min_prob = prob
                cur_min_path = trellis[tag][1]

        return cur_min_path
        
    def get_log_prob(self, dist, given, k):
        p = dist[given][k]
        if p > 0:
            return -math.log(p)
        else:
            return float('inf')

    def train(self):
        """Utilize the forward backward algorithm to train the HMM."""

        observation_sequences = []
        with open (self.trainpath ,"r") as infile: 
            for line in infile:
                words = []
                for wordtag in line.rstrip().split(" "):
                    if wordtag == "":
                        continue
                    # note that you might have escaped slashes
                    # 1\/2/CD means "1/2" "CD"
                    # keep 1/2 as 1\/2 
                    parts=wordtag.split("/")
                    tag=parts.pop()
                    word="/".join(parts)
                    words.append(word)
                observation_sequences.append(words)
        for _ in range(10):
            for observation in observation_sequences:
                alphas = self._forward(observation)
                betas = self._backward(observation)
                self._compute_new_params(alphas, betas, observation) 

