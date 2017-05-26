# Smoothen translation table given embeddings, from which cosine similarities are taken.
# sjmielke@jhu.edu

# Run in folder with:
# * polyglot-de.full.txt
# * polyglot-de.pkl
# * polyglot-en.full.txt
# * polyglot-en.pkl
# * lexicon.f2e.swap.de-en.200000.pruned-a+e-n5
# Creates some pickles and a subdir "runs" with the current iteration run as a subdir.

import numpy as np
import scipy.spatial.distance
from collections import defaultdict, Counter

import time
import pickle
import shutil
import os
import os.path
import sys

lexname = "lexicon.f2e.swap.de-en.200000.pruned-a+e-n5"
presoftmax_multiplier = 80.0
softmax = True

cheat = True
smooth_src = True
smooth_trg = False
pt_export_threshold = 0.0001
iters = 30
vocabsize = 20000

cheat = sys.argv[1] == "cheat"
smooth_src = sys.argv[2] == "src"
smooth_trg = sys.argv[3] == "trg"
pt_export_threshold = float(sys.argv[4])
iters = int(sys.argv[5])
vocabsize = int(sys.argv[6])

def get_similarity_matrix(lang):
    if os.path.isfile("polyglot-"+lang+"-cosmatrix-"+str(vocabsize)+".pickle"):
        with open("polyglot-"+lang+"-cosmatrix-"+str(vocabsize)+".pickle", 'rb') as f:
            (X_labels, cosmatrix) = pickle.load(f)
    else:
        if os.path.isfile("polyglot-"+lang+"-embmodel-"+str(vocabsize)+".pickle"):
            with open("polyglot-"+lang+"-embmodel-"+str(vocabsize)+".pickle", 'rb') as f:
                (source_words, model) = pickle.load(f)
        else:
            if os.path.isfile("polyglot-"+lang+".wordcounts.pickle"):
                with open("polyglot-"+lang+".wordcounts.pickle", 'rb') as f:
                    casecounter = pickle.load(f)
            else:
                # Get raw word counts in polyglot corpus
                casecounter = defaultdict(Counter)
                with open("polyglot-"+lang+".full.txt", encoding='utf-8') as f:
                    for line in f.readlines():
                        for word in line.split():
                            casecounter[word.lower()][word] += 1
                with open("polyglot-"+lang+".wordcounts.pickle", 'wb') as f:
                    pickle.dump(casecounter, f)
            print("Got raw word counts.", file = sys.stderr)
            
            # Get list of lowercase-count-tuples
            tups = []
            for (lcw, d) in casecounter.items():
                tups.append((sum(d.values()), lcw, d))
            tups = [(w, d) for (s, w, d) in sorted(tups, reverse = True)[0:vocabsize]]
            vocab_set = set([w for (w, d) in tups])
            print("Got overall word counts.", file = sys.stderr)
            
            # Get new dictionary of relative weights
            caseweighter = defaultdict(dict)
            for (lowerword, d) in casecounter.items():
                if lowerword in vocab_set:
                    Z = float(sum(d.values()))
                    for (caseword, c) in d.items():
                        caseweighter[lowerword][caseword] = float(c) / Z
            print("Got raw word counts.", file = sys.stderr)
            
            source_words = set()
            cased_model = {}
            dims = None
            with open("polyglot-"+lang+".pkl", 'rb') as f:
                o = pickle.load(f, encoding='latin-1')
                for (w, e) in zip(o[0], o[1]):
                    if w.lower() in vocab_set:
                        source_words.add(w)
                        cased_model[w] = e
                        assert(dims == None or dims == len(e))
                        dims = len(e)
            print("Got cased model.", file = sys.stderr)
            
            # Weighted average between casings
            model = {}
            for (lowerword, d) in caseweighter.items():
                indivs = [cased_model[casedword] * weight for (casedword, weight) in d.items() if casedword in cased_model]
                if indivs != []:
                    model[lowerword] = np.sum(indivs, axis = 0)
                    assert(len(model[lowerword]) == len(indivs[0]))
                else:
                    try:
                        print("No entry for »{}«".format(lowerword))
                    except UnicodeEncodeError:
                        print("No entry for a word containing unicode (damn this cluster)")
            print("Got lowercased model.", file = sys.stderr)
            
            print("Got all polyglot!", file = sys.stderr)
            
            with open("polyglot-"+lang+"-embmodel-"+str(vocabsize)+".pickle", 'wb') as f:
                pickle.dump((source_words, model), f)

        X_labels = sorted(list(model.keys()))
        X_vals   = np.array([model[w] for w in X_labels])

        cosmatrix = 1.0 - scipy.spatial.distance.pdist(X_vals, metric='cosine')
        cosmatrix = scipy.spatial.distance.squareform(cosmatrix)
        cosmatrix += np.eye(len(X_labels)) # dunno why this is necessary...

        # Cool. Save it for matrixfoo.py.
        with open("polyglot-"+lang+"-cosmatrix-"+str(vocabsize)+".pickle", 'wb') as f:
            pickle.dump((X_labels, cosmatrix), f)

    if softmax:
        simmatrix = np.zeros_like(cosmatrix)
        cosmatrix *= presoftmax_multiplier
        for i in range(len(cosmatrix)):
            Z = np.sum(np.exp(cosmatrix[i]))
            simmatrix[i] = np.exp(cosmatrix[i]) / Z
    else:
        simmatrix = cosmatrix
    
    print("Got similarities for", len(X_labels), lang, "embeddings!")
    
    return (X_labels, simmatrix)

def get_translation_matrix(X_labels_src, X_labels_trg):
    # populate vocab dicts if necessary
    srcdict_rev = {}
    trgdict_rev = {}
    srcdict = {l: i for (i, l) in enumerate(X_labels_src)}
    trgdict = {l: i for (i, l) in enumerate(X_labels_trg)}
    # construct matrix
    transmatrix = np.zeros((len(srcdict), len(trgdict)))
    with open(lexname, 'r', encoding='utf-8') as f:
        accept = 0
        discard = 0
        for line in f.read().splitlines():
            l = line.split(" ||| ")
            source = l[0]
            target = l[1]
            score = l[2]
            if source in srcdict and target in trgdict:
                transmatrix[srcdict[source]][trgdict[target]] = score if cheat else 1.0
                accept += 1
            else:
                #print("Discarded line:", line)
                discard += 1
    print("Read", accept, "useful lines and", discard, "lines whose source was not accepted.")
    return transmatrix

def export_phrase_table(filename, X_labels_src, trgdict_rev, translation_matrix):
    with open(filename, 'w', encoding='utf-8') as f:
        for (w1, transrow) in zip(X_labels_src, translation_matrix):
            for (i, score) in enumerate(transrow):
                if score > pt_export_threshold:
                    w2 = trgdict_rev[i]
                    print(w1, "|||", w2, "||| {:.7f} ||| ||| ".format(score), file = f)

# Get similarity matrix
(X_labels_src, src_simmatrix) = get_similarity_matrix("de")
(X_labels_trg, trg_simmatrix) = get_similarity_matrix("en")

# Get translation matrix
transmatrix = get_translation_matrix(X_labels_src, X_labels_trg)

revdict_trg = {i: w for (i, w) in enumerate(X_labels_trg)}

def translatable_stats(X_labels, transmatrix):
    words_okay = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row) > 0]
    words_nope = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row) == 0]
    res = "{:.2f}% translatable, {:.2f}% untranslatable\n".format(100.0 * len(words_okay) / len(X_labels), 100.0 * len(words_nope) / len(X_labels))
    words_okay = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row > pt_export_threshold) > 0]
    words_nope = [w for w, row in zip(X_labels, transmatrix) if np.count_nonzero(row > pt_export_threshold) == 0]
    pt_export_threshold
    res += "{:.2f}% in phrasetab, {:.2f}% not in phrasetab".format(100.0 * len(words_okay) / len(X_labels), 100.0 * len(words_nope) / len(X_labels))
    return res

dirname = "runs/" \
    + ("cheat_" if cheat else "") \
    + ("softmax-{}_".format(presoftmax_multiplier) if softmax else "") \
    + "vocab-{}_".format(vocabsize) \
    + "prune-below-{}_".format(pt_export_threshold) \
    + ("src" if smooth_src and not smooth_trg else ("trg" if not smooth_src and smooth_trg else "src-trg"))

if not os.path.isdir("runs"):
    os.mkdir("runs")

if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.mkdir(dirname)

"""
0) T := T = T0
1) T := Ss * T0 * Tt = T1
   Ss = Ss^2 = Ss2
   St = St^2 = St2
2) T := Ss2 * T0 * Tt2 = T2
   Ss = Ss2^2 = Ss4
   St = St2^2 = St4
3) T := Ss4 * T0 * Tt4 = T4
   Ss = Ss4^2 = Ss8
   St = St4^2 = Ss8
4) T := Ss8 * T0 * Tt8 = T8
   Ss = Ss8^2 = Ss16
   St = St8^2 = St16
"""

with open(dirname + "/iterations.log", 'w', encoding='utf-8') as lf:
    # Save it (sanity check)
    export_phrase_table(dirname + "/" + lexname + ".smoothed.0", X_labels_src, revdict_trg, transmatrix)
    print(translatable_stats(X_labels_src, transmatrix), flush = True)
    print(translatable_stats(X_labels_src, transmatrix), file = lf, flush = True)

    for i in range(iters):
        print("Iter {}: ".format(2**i), end='', flush = True)
        print("Iter {}: ".format(2**i), end='', file = lf, flush = True)
        start = time.time()
        # Calculate current translation matrix
        transmatrix_iter = transmatrix
        if smooth_src:
            transmatrix_iter = np.dot(src_simmatrix, transmatrix_iter)
        if smooth_trg:
            transmatrix_iter = np.dot(transmatrix_iter, trg_simmatrix)
        # Square similarity matrices!
        src_simmatrix = np.dot(src_simmatrix, src_simmatrix)
        trg_simmatrix = np.dot(trg_simmatrix, trg_simmatrix)
        # Give output
        end = time.time()
        export_phrase_table(dirname + "/" + lexname + ".smoothed.{}".format(2**i), X_labels_src, revdict_trg, transmatrix_iter)
        print(transmatrix_iter.sum(), "({} s)".format(end - start), flush = True)
        print(transmatrix_iter.sum(), "({} s)".format(end - start), file = lf, flush = True)
        print(translatable_stats(X_labels_src, transmatrix_iter), flush = True)
        print(translatable_stats(X_labels_src, transmatrix_iter), file = lf, flush = True)
