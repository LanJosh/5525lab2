from hmm import HiddenMarkovModel
h = HiddenMarkovModel(supervised=True)
x = h.eval('./pos_test.txt')
print(x)
