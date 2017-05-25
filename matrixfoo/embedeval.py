# Crunching embeddings for a big cosine similarity matrix
# sjmielke@jhu.edu

# Run in folder with
# * polyglot-de.full.txt
# * polyglot-de.pkl
# Leaves a couple pickles for the next script (matrixfoo.py) in there.

import numpy as np
import scipy.spatial.distance
from collections import defaultdict, Counter

import os.path
import time
import sys
import pickle

def import_polyglot():
	# Get raw word counts in polyglot corpus
	if os.path.isfile("polyglot-de.wordcounts.pickle"):
		with open("polyglot-de.wordcounts.pickle", 'rb') as f:
			casecounter = pickle.load(f)
	else:
		casecounter = defaultdict(Counter)
		with open("polyglot-de.full.txt", encoding='utf-8') as f:
			for line in f.readlines():
				for word in line.split():
					casecounter[word.lower()][word] += 1
		with open("polyglot-de.wordcounts.pickle", 'wb') as f:
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
	with open("polyglot-de.pkl", 'rb') as f:
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
	return (source_words, model)

def printnns(w):
	if w in model:
		print("»", w)
		w1 = model[w]
		cands = []
		for w2w in model.keys():
			w2 = model[w2w]
			#cands.append((1.0 - scipy.spatial.distance.euclidean(w1, w2), w2w))
			cands.append((1.0 - scipy.spatial.distance.cosine(w1, w2), w2w))
		l = sorted(cands, reverse=True)[:5]
		print("\n".join(["{:4.4f} {}".format(s, w) for (s,w) in l]))

if os.path.isfile("polyglot-de-embmodel.pickle"):
	with open("polyglot-de-embmodel.pickle", 'rb') as f:
		(source_words, model) = pickle.load(f)
else:
	(source_words, model) = import_polyglot()
	with open("polyglot-de-embmodel.pickle", 'wb') as f:
		pickle.dump((source_words, model), f)

# printnns('deutschland')

X_labels = sorted(list(model.keys()))
X_vals   = np.array([model[w] for w in X_labels])

cosmatrix = 1.0 - scipy.spatial.distance.pdist(X_vals, metric='cosine')
cosmatrix = scipy.spatial.distance.squareform(cosmatrix)
cosmatrix += np.eye(len(X_labels)) # dunno why this is necessary...

# print("» deutschland")
# print(cosmatrix[X_labels.index("deutschland")])
# deu_sims = list(cosmatrix[X_labels.index("deutschland")])
# for (s, w) in sorted(list(zip(deu_sims, X_labels)), reverse = True)[0:5]:
# 	print("{:.4f}".format(s), w)

# Cool. Save it for matrixfoo.py.
with open("polyglot-de-cosmatrix.pickle", 'wb') as f:
	pickle.dump((X_labels, cosmatrix), f)
