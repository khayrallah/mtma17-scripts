import random
import itertools
import time
from collections import defaultdict, deque

string_size = 9
vocab_size = 1000

# https://stackoverflow.com/a/6822761
def windows(seq, n=2):
    it = iter(seq)
    win = deque((next(it, None) for _ in range(n)), maxlen=n)
    yield tuple(win)
    append = win.append
    for e in it:
        append(e)
        yield tuple(win)

# Given a string of length 10
s = [random.randint(0, vocab_size - 1) for _ in range(string_size)]

def get_lm():
    # Given a LM_target for triples or whatever (assume log probs)
    lm = defaultdict(lambda: random.random())

    for w1 in range(vocab_size):
        for w2 in range(vocab_size):
            lm[(w1, w2)]
            #print(f"{w1} {w2} # {lm[(w1, w2)]:.3f}")

    return dict(lm)

def getit_bruteforce(lm):
    # get all permutations
    i = 0
    bestscore = 0.0
    bestperm = None
    for perm in itertools.permutations(s):
        i += 1
        score = sum([lm[window] for window in windows(perm, 2)])
        if score > bestscore:
            bestscore = score
            bestperm = perm
    return (bestperm, bestscore)

def getit_dynprog(lm):
    # Pick one of the string
    for startingindex in range(len(s)):
        

lm = get_lm()
start = time.time()
(bestperm, bestscore) = getit_bruteforce(lm)
print(bestperm, '#', bestscore, f" ({time.time() - start:.2f}s)")

