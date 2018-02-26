from collections import defaultdict
from collections import Counter

import math


class HiddenMarkovModel:
    """HMM for the Eisner icecream data
    """

    def __init__(
        self, train="./pos_train.txt", supervised=True,wordCase='regular'):
        """
        Args:
            train: str. The path to the file containing the training data.
            supervised: bool. Whether or not to use the tags in the part of speech
                data.
        """

        self.epsilon = 0.000001
        self.trainpath=train

        self.states = ['H', 'C']
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
        if not supervised:
            self._setup_bw()

    def _setup_bw(self):
        self.chi = defaultdict(lambda: defaultdict(float))
        self.total_chi = defaultdict(float)
        self.seen_observations = set()

        self.gamma = defaultdict(float)
        self.gamma_by_obs = defaultdict(lambda: defaultdict(float))
        self.start_gammas = defaultdict(float)
        self.end_gammas = defaultdict(float)


    def _forward(self, observations):
        """Forward step of training the HMM.
        Args:
            observations: A list of strings.
        Returns:
            A list of dict representing the trellis of alpha values
        """
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

    def _expectation(self, alphas, betas, observations):
        total_sent_prob = alphas[-1]['</s>']
        if total_sent_prob == 0:
            total_sent_prob = self.epsilon

        # E-step
        for t in range(len(observations)):
            self.seen_observations.add(observations[t])
            for state in self.states:
                if t != 0:
                    for next_state in self.states:
                        p = alphas[t-1][state] * self.trans_prob[state][next_state] * \
                            self.obs_prob[next_state][observations[t]] * \
                            betas[t + 1][next_state] / total_sent_prob
                        self.total_chi[state] += p
                        self.chi[state][next_state] += p


                p = alphas[t][state] * betas[t + 1][state] / total_sent_prob

                if t == 0:
                    self.start_gammas[state] += p
                if t == len(observations) - 1:
                    self.end_gammas[state] += p
                self.gamma[state] += p
                self.gamma_by_obs[state][observations[t]] += p


        for state in self.states:
            self.total_chi[state] += alphas[-2][state] * self.trans_prob[state]['</s>'] / total_sent_prob


    def _maximization(self):
        for i in self.states:
            total_chi_prob = self.total_chi[i]
            if total_chi_prob == 0:
                total_chi_prob = self.epsilon

            for j in self.states:
                self.trans_prob[i][j] = self.chi[i][j] / total_chi_prob

            for v_k in self.seen_observations:
                state_prob = self.gamma[i]
                if state_prob == 0:
                    state_prob = self.epsilon
                self.obs_prob[i][v_k] = self.gamma_by_obs[i][v_k] / state_prob

        for i in self.states:
            state_prob = self.gamma[i]
            if state_prob == 0:
                state_prob = self.epsilon

            self.trans_prob['<s>'][i] = self.start_gammas[i]
            self.trans_prob[i]['</s>'] = self.end_gammas[i] / state_prob


    def viterbi(self, words):
        trellis = {}
        for tag in self.states:
            trellis[tag] = [self.get_log_prob(self.trans_prob, '<s>', tag), [tag]]
            if words[0] in self.vocabulary:
                trellis[tag][0] += self.get_log_prob(self.obs_prob, tag, words[0])
            else:
                trellis[tag][0] += self.get_log_prob(self.obs_prob, tag, '<UNK>')

        new_trellis = {}
        for word in words[1:]:
            for cur_tag in self.states:
                cur_min_prob = float('inf')
                cur_min_path = None

                for prev_tag in self.states:
                    prob = trellis[prev_tag][0] + self.get_log_prob(self.trans_prob, prev_tag, cur_tag)
                    if word in self.vocabulary:
                        prob += self.get_log_prob(self.obs_prob, cur_tag, word)
                    else:
                        prob += self.get_log_prob(self.obs_prob, cur_tag, '<UNK>')

                    if prob <= cur_min_prob:
                        cur_min_prob = prob
                        cur_min_path = trellis[prev_tag][1] + [cur_tag]

                new_trellis[cur_tag] = [cur_min_prob, cur_min_path]

            trellis = new_trellis
            new_trellis = {}

        cur_min_prob = float('inf')
        cur_min_path = None
        for tag in self.states:
            prob = self.get_log_prob(self.trans_prob, tag, '</s>') + trellis[tag][0]
            if prob <= cur_min_prob:
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
        for x in range(13):
            print("Iteration {}".format(x))
            for state in self.states:
                print('P({}|{}) = {}'.format(state, '<s>', self.trans_prob['<s>'][state]))
                print('P({}|{}) = {}'.format('</s>', state, self.trans_prob[state]['</s>']))
            for state in self.states:
                for state2 in self.states:
                    print('P({}|{}) = {}'.format(state, state2, self.trans_prob[state2][state]))
            for observation in ['1','2','3']:
                for state in self.states:
                    print('P({}|{}) = {}'.format(observation, state, self.obs_prob[state][observation]))

            alphas = self._forward(self.observations)
            betas = self._backward(self.observations)
            self._expectation(alphas, betas, self.observations) 
            self._maximization()
            self._setup_bw()

    def eval(self, testpath,wordCase='regular'):
        correct = 0
        total = 0
        with open(testpath, 'r') as testf:
            for i, line in enumerate(testf):
                line = line.strip()
                terms = line.split()

                tokens = []
                tags = []
                for term in terms:
                    slash_idx = term.rindex('/')
                    token, tag = term[:slash_idx], term[slash_idx + 1:]
                    if wordCase=='regular':
                        tokens.append(token)
                    else:
                        tokens.append(token.lower())
                        
                    tags.append(tag)

                predicted_tags = self.viterbi(tokens)
                for predicted_tag, actual_tag in zip(predicted_tags, tags):
                    total += 1
                    if predicted_tag == actual_tag:
                        correct += 1

        return correct / total
