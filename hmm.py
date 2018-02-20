from collections import Counter


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

                    lasttag=tag
        # Compute the probability matrices A and B
        self.alpha = {}
        for tag1 in self.tag_given_tag_counts.keys():
            norm = sum(self.tag_given_tag_counts[tag].values())
            for tag2 in self.tag_given_tag_counts.keys():
                self.alpha[tag1+tag2] = self.tag_given_tag_counts[tag1][tag2] / norm

    def _forward(self):
        """Forward step of training the HMM."""
        

    def _backward(self):
        """Backward step of training the HMM."""

    def viterbi(self):
        """Viterbi for decoding the HMM."""
        

    def train(self, iteration=10):
        """Utilize the forward backward algorithm to train the HMM."""

