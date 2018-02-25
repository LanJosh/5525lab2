from collections import defaultdict
from collections import Counter

import math
import sys


class HiddenMarkovModel:
    """HMM for the Eisner icecream data
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

        self.states = set(['C', 'H'])
        self.observations = list(map(str, [2,3,3,2,3,2,3,2,2,3,1,3,3,1,1,1,2,1,1,1,3,1,2,1,1,1,2,3,3,2,3,2,2]))

        # Compute the probability matrices A and B
        self.trans_prob = defaultdict(lambda: defaultdict(float)) # A
        self.trans_prob['C']['C'] = .8
        self.trans_prob['H']['C'] = .1
        self.trans_prob['<s>']['C'] = .5
        self.trans_prob['C']['H'] = .1
        self.trans_prob['H']['H'] = .8
        self.trans_prob['<s>']['H'] = .5
        self.trans_prob['C']['</s>'] = .1
        self.trans_prob['H']['</s>'] = .1
        self.trans_prob['<s>']['</s>'] = .0

        self.obs_prob = defaultdict(lambda: defaultdict(float)) # B
        self.obs_prob['C']['1'] = .7
        self.obs_prob['H']['1'] = .1
        self.obs_prob['C']['2'] = .2
        self.obs_prob['H']['2'] = .2
        self.obs_prob['C']['3'] = .1
        self.obs_prob['H']['3'] = .7

    def _forward(self, observations):
        """Forward step of training the HMM.

        Args:
            observations: A list of strings.

        Returns:
            A list of dict representing the trellis of alpha values
        """
        observations = self.observations
        states = self.states
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

        trellis.append({})
        trellis[-1]['</s>'] = sum(trellis[-2][s] * self.trans_prob[s]['</s>'] for s in self.states)

        return trellis

    def _backward(self, observations):
        """Backward step of training the HMM.

        Args:
            observations: A list of strings.

        Returns:
            A list of dict representing the trellis of beta values
        """
        states = self.states
        observations = self.observations
        trellis = [{}]

        for state in states:
            trellis[0][state] = self.trans_prob[state]['</s>']

        for t in range(len(observations)-1, 0, -1):
            trellis.insert(0, {})
            for state in states:
                trellis[0][state] = sum(trellis[1][next_state]
                    * self.trans_prob[state][next_state]
                    * self.obs_prob[next_state][observations[t]]
                    for next_state in states)

        trellis.insert(0, {})
        trellis[0]['<s>'] = sum(trellis[1][s] *
                                self.trans_prob['<s>'][s] *
                                self.obs_prob[s][observations[0]] for s in self.states)

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
        i = 0
        for t in range(1,len(observations),1):
            chi.append(defaultdict(lambda: defaultdict(float)))
            for state in self.states:
                for next_state in self.states:
                    # Using chi defined in the Eisner sheet
                    # being in state `next_state` at time t having transitioned from state `state` at 
                    # t-1. Jurafsky & Martin defined it as being in state `state` at time t and 
                    # going to state `next_state` at t+1. The denominator is also different in definition
                    # but both represent the probability of the observation sequence.
                    chi[t-1][state][next_state] = (alphas[t-1][state] * self.trans_prob[state][next_state]
                        * self.obs_prob[next_state][observations[t]] * betas[t + 1][next_state] / alphas[-1]['</s>'])

        gamma = []
        for t in range(len(observations)):
            gamma.append(defaultdict(float))
            for state in self.states:
                gamma[t][state] = alphas[t][state] * betas[t + 1][state] / alphas[-1]['</s>']

        # M-step
        for i in self.states:
            for j in self.states:
                total_prob = sum(chi[t][i][k] for t in range(len(observations)-1) for k in self.states) + \
                             alphas[-2][i] * self.trans_prob[i]['</s>'] / alphas[-1]['</s>']
                self.trans_prob[i][j] = (sum(chi[t][i][j] for t in range(len(observations)-1))) / total_prob

            for v_k in observations:
                self.obs_prob[i][v_k] = sum(gamma[t][i] for t in range(len(observations)) if observations[t] == v_k)
                self.obs_prob[i][v_k] /= sum(gamma[t][i] for t in range(len(observations)))

        for i in self.states:
            self.trans_prob['<s>'][i] = gamma[0][i]
            self.trans_prob[i]['</s>'] = gamma[-1][i] / sum(gamma[t][i] for t in range(len(observations)))

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

        for x in range(10):
            alphas = self._forward(self.observations)
            betas = self._backward(self.observations)
            self._compute_new_params(alphas, betas, self.observations) 

        for state in self.states:
            for state2 in self.states:
                print('P({}|{}) = {}'.format(state, state2, self.trans_prob[state2][state]))
        for observation in ['1','2','3']:
            for state in self.states:
                print('P({}|{}) = {}'.format(observation, state, self.obs_prob[state][observation]))

