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
        self.testpath=test
        self.trainpath=train

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

        # Compute the probability matrices A and B
        self.alpha = {}
        for tag1 in self.tag_given_tag_counts.keys():
            norm = sum(self.tag_given_tag_counts[tag1].values())
            for tag2 in self.tag_given_tag_counts[tag1].keys():
                self.alpha[tag2 + ' ' + tag1] = self.tag_given_tag_counts[tag1][tag2] / norm

        self.vocabulary = set()
        self.beta = {}
        for tag in self.word_given_tag_counts.keys():
            norm = sum(self.word_given_tag_counts[tag].values())
            for word in self.word_given_tag_counts[tag].keys():
                self.vocabulary.add(word)
                self.beta[word + ' ' + tag] = self.word_given_tag_counts[tag][word] / norm

    def _forward(self):
        """Forward step of training the HMM."""
        
    def _backward(self):
        """Backward step of training the HMM."""

    def viterbi(self, words):
        tags = self.tag_given_tag_counts.keys()
        vocabulary = self.vocabulary

        trellis = {}
        for tag in tags:
            if words[0] in vocabulary:
                trellis[tag] = (self.get_log_prob(self.alpha, tag + ' <s>') + \
                                self.get_log_prob(self.beta, words[0] + ' ' + tag), ['<s>'])
            else:
                trellis[tag] = (self.get_log_prob(self.alpha, tag + ' <s>') + \
                                self.get_log_prob(self.beta, '<UNK> ' + tag), ['<s>'])

        print(trellis)

        for word in words[1:]:
            for cur_tag in tags:
                cur_min_prob = float('inf')
                cur_min_path = None

                for prev_tag in tags:
                    if word in vocabulary:
                        prob = self.get_log_prob(self.alpha, cur_tag + ' ' + prev_tag) + \
                               self.get_log_prob(self.beta, word + ' ' + tag) + \
                               trellis[prev_tag][0]
                    else:
                        prob = self.get_log_prob(self.alpha, cur_tag + ' ' + prev_tag) + \
                               self.get_log_prob(self.beta, '<UNK> ' + tag) + \
                                trellis[prev_tag][0]

                    if prob < cur_min_prob:
                        print('here')
                        cur_min_prob = prob
                        cur_min_path = trellis[prev_tag][1] + [tag]

                trellis[cur_tag] = (cur_min_prob, cur_min_path)

        cur_min_prob = float('inf')
        cur_min_path = None
        for tag in tags:
            prob = self.get_log_prob(self.alpha, '</s> ' + tag) + trellis[tag][0]
            if prob < cur_min_prob:
                cur_min_prob = prob
                cur_min_path = trellis[tag][1] + ['</s>']

        trellis['</s>'] = (cur_min_prob, cur_min_path)

        min_prob = float('inf')
        min_tag_seq = None
        for k, v in trellis.items():
            if v[1] < min_prob:
                min_prob = v[0]
                min_tag_seq = v[1]

        return min_tag_seq
        
    def get_log_prob(self, dist, k):
        try:
            p = dist[k]
            return -math.log(p)
        except KeyError:
            return float('inf')

    def train(self, iteration=10):
        """Utilize the forward backward algorithm to train the HMM."""

