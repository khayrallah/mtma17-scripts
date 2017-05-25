import numpy as np
import time
import pickle
import shutil
import os
import os.path

def get_toy_matrices():
    vocab_size_s = 4

    wordmap_s = {"wörter": 0, "worte": 1, "wegen": 2, "weil": 3}
    wordmap_t = {"words": 0, "because": 1, "as": 2, "foo": 3}

    similarities_raw = {
        "wörter": [("wörter", 1.0), ("worte", 0.9), ("wegen", 0.0), ("weil", 0.0)],
        "worte":  [("wörter", 0.4), ("worte", 1.0), ("wegen", 0.0), ("weil", 0.0)],
        "wegen":  [("wörter", 0.0), ("worte", 0.0), ("wegen", 1.0), ("weil", 0.6)],
        "weil":   [("wörter", 0.0), ("worte", 0.0), ("wegen", 0.6), ("weil", 1.0)],
    }

    # We could assume symmetry, but... eh.
    similarity_matrix = np.zeros((vocab_size_s, vocab_size_s))

    for (w1, l) in similarities_raw.items():
        for (w2, score) in l:
            similarity_matrix[wordmap_s[w1], wordmap_s[w2]] = score

    # first axis: V_S, second axis: V_T
    translation_matrix = np.zeros((vocab_size_s, vocab_size_s))
    
    # Populate initial matrix
    with open("/home/sjm/documents/MTMA2017/mtma17-scripts/matrixfoo/miniphrasetable") as f:
        for line in f.read().splitlines():
            [source, target, score, _] = line.split(" ||| ")
            i_s = wordmap_s[source]
            i_t = wordmap_t[target]
            translation_matrix[i_s, i_t] = score
    
    return (translation_matrix, similarity_matrix)

def get_random_matrices(size):
    similarity_matrix = np.random.rand(size, size)
    translation_matrix = np.random.rand(size, size)
    return (translation_matrix, similarity_matrix)

def get_similarity_matrix():
    with open("../embedding-matrixfoo-files/polyglot-de-cosmatrix.pickle", 'rb') as f:
        (X_labels, cosmatrix) = pickle.load(f)
    return (X_labels, cosmatrix)

def get_translation_matrix(X_labels):
    srcdict = {l: i for (i, l) in enumerate(X_labels)}
    ptpath = "../embedding-matrixfoo-files/lexicon.e2f.de-en.200000.pruned-a+e-n3"
    # populate target vocab dict
    trgdict = {}
    trgdict_rev = {}
    with open(ptpath, 'r', encoding='utf-8') as f:
        i = 0
        for line in f.read().splitlines():
            l = line.split(" ||| ")
            target = l[1]
            if target not in trgdict:
                trgdict[target] = i
                trgdict_rev[i] = target
                i += 1
    # construct matrix
    transmatrix = np.zeros((len(srcdict), len(trgdict)))
    with open(ptpath, 'r', encoding='utf-8') as f:
        accept = 0
        discard = 0
        for line in f.read().splitlines():
            l = line.split(" ||| ")
            source = l[0]
            target = l[1]
            score = l[2]
            if source in srcdict:
                transmatrix[srcdict[source]][trgdict[target]] = score
                accept += 1
            else:
                #print("Discarded line:", line)
                discard += 1
    print(f"Read {accept} useful lines and {discard} lines whose source was not accepted.")
    return (trgdict_rev, transmatrix)

pt_export_threshold = 0.0001
presoftmax_multiplier = 80.0
softmax = True

def export_phrase_table(filename, X_labels, trgdict_rev, translation_matrix):
    with open(filename, 'w') as f:
        for (w1, transrow) in zip(X_labels, translation_matrix):
            for (i, score) in enumerate(transrow):
                if score > pt_export_threshold:
                    w2 = trgdict_rev[i]
                    print(f"{w1} ||| {w2} ||| {score:.7f} ||| ||| ", file = f)

# Get similarity matrix
(X_labels, cosmatrix) = get_similarity_matrix()
start = time.time()
# Softmax similarity matrix
if softmax:
    newmat = np.zeros_like(cosmatrix)
    cosmatrix *= presoftmax_multiplier
    for i in range(len(cosmatrix)):
        Z = np.sum(np.exp(cosmatrix[i]))
        newmat[i] = np.exp(cosmatrix[i]) / Z
    cosmatrix = newmat
end = time.time()
print(f"Similarity matrix softmaxing took {end - start} s")
# Get translation matrix
(trgdict_rev, transmatrix) = get_translation_matrix(X_labels)

def translatable_stats(X_labels, transmatrix):
    words_okay = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row) > 0]
    words_nope = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row) == 0]
    res = f"{100.0 * len(words_okay) / len(X_labels):.2f}% translatable, {100.0 * len(words_nope) / len(X_labels):2f}% untranslatable\n"
    words_okay = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row > pt_export_threshold) > 0]
    words_nope = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row > pt_export_threshold) == 0]
    pt_export_threshold
    res += f"{100.0 * len(words_okay) / len(X_labels):.2f}% in phrasetab, {100.0 * len(words_nope) / len(X_labels):2f}% not in phrasetab"
    return res


dirname = "../embfooruns/" + (f"softmax-{presoftmax_multiplier}-" if softmax else "") + f"prune-below-{pt_export_threshold}"

if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.mkdir(dirname)

with open(dirname + "/iterations.log", 'w') as lf:
    # Save it (sanity check)
    export_phrase_table(dirname + "/pt.0", X_labels, trgdict_rev, transmatrix)
    print(translatable_stats(X_labels, transmatrix))
    print(translatable_stats(X_labels, transmatrix), file = lf, flush = True)
    #print("Transmatrix sum:", transmatrix.sum())

    for i in range(30):
        print(f"Iter {i+1}: ", end='')
        print(f"Iter {i+1}: ", end='', file = lf, flush = True)
        start = time.time()
        transmatrix = np.dot(cosmatrix, transmatrix)
        #transmatrix /= transmatrix.sum()
        print(transmatrix.sum(), end='')
        #print(transmatrix)
        end = time.time()
        export_phrase_table(dirname + f"/pt.{i+1}", X_labels, trgdict_rev, transmatrix)
        print(f" ({end - start} s)")
        print(f" ({end - start} s)", file = lf, flush = True)
        print(translatable_stats(X_labels, transmatrix))
        print(translatable_stats(X_labels, transmatrix), file = lf, flush = True)
