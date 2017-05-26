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
smooth_src = "de"
smooth_trg = None
pt_export_threshold = 0.0001
iters = 30

cheat = sys.argv[1] == "cheat"
smooth_src = sys.argv[2] if sys.argv[2] != "None" else None
smooth_trg = sys.argv[3] if sys.argv[3] != "None" else None
pt_export_threshold = float(sys.argv[4])
iters = int(sys.argv[5])

def get_similarity_matrix(lang):
    if os.path.isfile("polyglot-"+lang+"-cosmatrix.pickle"):
        with open("polyglot-"+lang+"-cosmatrix.pickle", 'rb') as f:
            (X_labels, cosmatrix) = pickle.load(f)
    else:
        if os.path.isfile("polyglot-"+lang+"-embmodel.pickle"):
            with open("polyglot-"+lang+"-embmodel.pickle", 'rb') as f:
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
            tups = [(w, d) for (s, w, d) in sorted(tups, reverse = True)[0:20000]]
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
            
            with open("polyglot-"+lang+"-embmodel.pickle", 'wb') as f:
                pickle.dump((source_words, model), f)

        X_labels = sorted(list(model.keys()))
        X_vals   = np.array([model[w] for w in X_labels])

        cosmatrix = 1.0 - scipy.spatial.distance.pdist(X_vals, metric='cosine')
        cosmatrix = scipy.spatial.distance.squareform(cosmatrix)
        cosmatrix += np.eye(len(X_labels)) # dunno why this is necessary...

        # Cool. Save it for matrixfoo.py.
        with open("polyglot-"+lang+"-cosmatrix.pickle", 'wb') as f:
            pickle.dump((X_labels, cosmatrix), f)

    if softmax:
        simmatrix = np.zeros_like(cosmatrix)
        cosmatrix *= presoftmax_multiplier
        for i in range(len(cosmatrix)):
            Z = np.sum(np.exp(cosmatrix[i]))
            simmatrix[i] = np.exp(cosmatrix[i]) / Z
    else:
        simmatrix = cosmatrix
    
    return (X_labels, simmatrix)

def get_translation_matrix(X_labels_src, X_labels_trg):
    # populate vocab dicts if necessary
    srcdict = {}
    srcdict_rev = {}
    trgdict = {}
    trgdict_rev = {}
    if X_labels_src:
        srcdict = {l: i for (i, l) in enumerate(X_labels_src)}
    if X_labels_trg:
        trgdict = {l: i for (i, l) in enumerate(X_labels_trg)}
    with open(lexname, 'r', encoding='utf-8') as f:
        i = 0
        j = 0
        for line in f.read().splitlines():
            l = line.split(" ||| ")
            source = l[0]
            target = l[1]
            if not X_labels_src and source not in srcdict:
                srcdict[source] = i
                srcdict_rev[i] = source
                i += 1
            if not X_labels_trg and target not in trgdict:
                trgdict[target] = j
                trgdict_rev[j] = target
                j += 1
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
    return (srcdict_rev, trgdict_rev, transmatrix)

def export_phrase_table(filename, X_labels_src, trgdict_rev, translation_matrix):
    with open(filename, 'w', encoding='utf-8') as f:
        for (w1, transrow) in zip(X_labels_src, translation_matrix):
            for (i, score) in enumerate(transrow):
                if score > pt_export_threshold:
                    w2 = trgdict_rev[i]
                    print(w1, "|||", w2, "||| {:.7f} ||| ||| ".format(score), file = f)

# Get similarity matrix
X_labels_src = None
X_labels_trg = None
if smooth_src:
    (X_labels_src, src_simmatrix) = get_similarity_matrix(smooth_src)
if smooth_trg:
    (X_labels_trg, trg_simmatrix) = get_similarity_matrix(smooth_trg)

# Get translation matrix
(revdict_src, revdict_trg, transmatrix) = get_translation_matrix(X_labels_src, X_labels_trg)

if not smooth_src:
    X_labels_src = [w for (i, w) in sorted(list(revdict_src.items()))]

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
    + "prune-below-{}_".format(pt_export_threshold) \
    + ("src" if smooth_src and not smooth_trg else ("trg" if not smooth_src and smooth_trg else "src-trg"))

if not os.path.isdir("runs"):
    os.mkdir("runs")

if os.path.isdir(dirname):
    shutil.rmtree(dirname)
os.mkdir(dirname)

with open(dirname + "/iterations.log", 'w', encoding='utf-8') as lf:
    # Save it (sanity check)
    export_phrase_table(dirname + "/" + lexname + ".smoothed.0", X_labels_src, revdict_trg, transmatrix)
    print(translatable_stats(X_labels_src, transmatrix))
    print(translatable_stats(X_labels_src, transmatrix), file = lf, flush = True)
    #print("Transmatrix sum:", transmatrix.sum())

    for i in range(iters):
        print("Iter {}: ".format(i+1), end='')
        print("Iter {}: ".format(i+1), end='', file = lf, flush = True)
        start = time.time()
        if smooth_src:
            transmatrix = np.dot(src_simmatrix, transmatrix)
        if smooth_trg:
            transmatrix = np.dot(transmatrix, trg_simmatrix)
        #transmatrix /= transmatrix.sum()
        print(transmatrix.sum(), end='')
        #print(transmatrix)
        end = time.time()
        export_phrase_table(dirname + "/" + lexname + ".smoothed.{}".format(i+1), X_labels_src, revdict_trg, transmatrix)
        print(" ({} s)".format(end - start))
        print(" ({} s)".format(end - start), file = lf, flush = True)
        print(translatable_stats(X_labels_src, transmatrix))
        print(translatable_stats(X_labels_src, transmatrix), file = lf, flush = True)
