from hmm import HiddenMarkovModel
h = HiddenMarkovModel(supervised=False)
h.train()
x = h.eval('./pos_test.txt')
print(x)
