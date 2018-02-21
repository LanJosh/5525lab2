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
        self.word_counts=dict()

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
                    if word not in self.word_counts:
                        self.word_counts[word]=1
                    self.word_counts[word]+=1

                    lasttag=tag
                if lasttag not in self.tag_given_tag_counts:
                    self.tag_given_tag_counts[lasttag] = Counter()
                self.tag_given_tag_counts[lasttag]["</s>"]+=1

        self.mode_tag = tag_counts.most_common(1)[0][0]

        # Compute the probability matrices A and B
        self.alpha = defaultdict(lambda: defaultdict(int))
        for tag1 in self.tag_given_tag_counts.keys():
            norm = sum(self.tag_given_tag_counts[tag1].values())
            for tag2, count in self.tag_given_tag_counts[tag1].items():
                self.alpha[tag1][tag2] = count / norm


        self.vocabulary = set()
        self.beta = defaultdict(lambda: defaultdict(int))
        smooth_adj = self.epsilon * len(self.word_given_tag_counts[self.mode_tag])

        for tag in self.word_given_tag_counts.keys():
            norm = sum(self.word_given_tag_counts[tag].values())
            if tag == self.mode_tag:
                adj = smooth_adj
                self.beta[self.mode_tag]['<UNK>'] = self.epsilon / (norm + adj)
            else:
                adj = 0

            for word, count in self.word_given_tag_counts[tag].items():
                self.vocabulary.add(word)
                self.beta[tag][word] = (count + self.epsilon) / (norm + adj)


        self.tags = self.tag_given_tag_counts.keys()

    def _forward(self):
        """Forward step of training the HMM."""
        
    def _backward(self):
        """Backward step of training the HMM."""

    def viterbi(self, words):
        trellis = {}
        for tag in self.tags:
            trellis[tag] = [self.get_log_prob(self.alpha, '<s>', tag), ['<s>', tag]]
            if words[0] in self.vocabulary:
                trellis[tag][0] += self.get_log_prob(self.beta, tag, words[0])
            else:
                trellis[tag] += self.get_log_prob(self.beta, tag, '<UNK>')

        new_trellis = {}
        for word in words[1:]:
            for cur_tag in self.tags:
                cur_min_prob = float('inf')
                cur_min_path = None

                for prev_tag in self.tags:
                    prob = trellis[prev_tag][0] + self.get_log_prob(self.alpha, prev_tag, cur_tag)
                    if word in self.vocabulary:
                        prob += self.get_log_prob(self.beta, cur_tag, word)
                    else:
                        prob += self.get_log_prob(self.beta, cur_tag, '<UNK>')

                    if prob < cur_min_prob:
                        cur_min_prob = prob
                        cur_min_path = trellis[prev_tag][1] + [cur_tag]

                new_trellis[cur_tag] = [cur_min_prob, cur_min_path]

            trellis = new_trellis
            new_trellis = {}

        cur_min_prob = float('inf')
        cur_min_path = None
        for tag in self.tags:
            prob = self.get_log_prob(self.alpha, tag, '</s>') + trellis[tag][0]
            if prob < cur_min_prob:
                cur_min_prob = prob
                cur_min_path = trellis[tag][1] + ['</s>']

        trellis['</s>'] = [cur_min_prob, cur_min_path]

        return trellis['</s>'][1]
        
    def get_log_prob(self, dist, given, k):
        p = dist[given][k]
        if p > 0:
            return -math.log(p)
        else:
            return float('inf')

    def train(self, iteration=10):
        """Utilize the forward backward algorithm to train the HMM."""

