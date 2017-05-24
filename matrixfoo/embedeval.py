from lshash.lshash import LSHash
from bitarray import bitarray
from math import cos
import numpy as np
import scipy.spatial.distance

import os.path
import time
import sys
import pickle

def data_preprocessing(vocab_size = 100000):
	with open('/mnt/hylia/embed_wiki_de_2M_200D/vocabulary.pickle', 'rb') as f:
		o = pickle.load(f)
		source_words = set([w for (s, w) in sorted([(s, w) for (w,s) in o.items()])][0:vocab_size])
	print("Got vocab.", file = sys.stderr)

	model = {}
	i = 0
	with open("/mnt/hylia/embed_wiki_de_2M_200D/embedding_file") as f:
		l = f.readline()
		[vocabsize, dims] = l.split()
		[vocabsize, dims] = [int(vocabsize), int(dims)]
		l = f.readline()
		while l != '':
			l = l[:-1] # remove trailing newline
			l = l.split()
			w = l[0]
			if w in source_words:
				vals = np.array([float(v) for v in l[1:]])
				assert(len(vals) == dims)
				model[w] = vals
			l = f.readline() # get next one
			i += 1
	assert(len(model) == vocab_size)
	print("Got embeddings.", file = sys.stderr)

	planes = 10
	lsh = LSHash(planes, dims)
	for w in source_words:
		lsh.index(model[w], extra_data=w)
	#source_hashes = [bitarray(lsh._hash(lsh.uniform_planes[0], model[w])) for w in source_words]
	print("Got hashes.", file = sys.stderr)
	
	return (source_words, model, lsh)

def nns_hash(w):
	return [(sim, word) for ((vals, word), sim) in lsh.query(model[w], distance_func = "cosine")][0:5]

def nns_truecosin(w):
	w1 = model[w]
	cands = []
	for w2w in source_words:
		w2 = model[w2w]
		#cands.append((1.0 - scipy.spatial.distance.euclidean(w1, w2), w2w))
		cands.append((1.0 - scipy.spatial.distance.cosine(w1, w2), w2w))
	return sorted(cands, reverse=True)[:5]

def printnns(w):
	print(f"» {w}")
	l = nns_truecosin(w)
	print("\n".join([f"{s:4.4f} {w}" for (s,w) in l]))



if os.path.isfile("/tmp/embmodel.pickle"):
	with open("/tmp/embmodel.pickle", 'rb') as f:
		(source_words, model, lsh) = pickle.load(f)
else:
	(source_words, model, lsh) = data_preprocessing()
	with open("/tmp/embmodel.pickle", 'wb') as f:
		pickle.dump((source_words, model, lsh), f)


printnns('frosch')
printnns('deutschland')
printnns('deutsch')
printnns('dann')
printnns('blau')
printnns('weg')
printnns('straße')
exit(0)

def simi(w1, w2):
	if w1 not in model or w2 not in model:
		print(f"{w1} or {w2} not in model")
	else:
		h1 = bitarray(lsh._hash(lsh.uniform_planes[0], model[w1]))
		h2 = bitarray(lsh._hash(lsh.uniform_planes[0], model[w2]))
		s_h = cos(3.141592653589 * float((h1 ^ h2).count()) / float(planes))
		s_w = 1 - cosine(model[w1], model[w2])
		print(f"{w1} {w2} = {s_h:.4f} hash {s_w:.4f} word => {abs(s_h-s_w):.4f} diff")

simi("obwohl", "trotzdem")
simi("denn", "weil")
simi("dann", "danach")

simi("tasse", "becher")
simi("brot", "brötchen")
simi("schwarzbrot", "graubrot")
simi("kessel", "topf")

simi("pfad", "straße")
simi("könig", "kaiser")
simi("könig", "führer")
simi("blau", "grün")
simi("blau", "rot")
simi("blau", "gelb")
simi("blau", "scharf")
simi("blau", "stumpf")

print("Starting measurements")
for i in range(1):
	start = time.time()
	for (w1, h1) in zip(source_words, source_hashes):
		for (w2, h2) in zip(source_words, source_hashes):
			#s_w = 1 - cosine(model[w1], model[w2])
			s_h_form = cos(3.141592653589 * float((h1 ^ h2).count()) / float(planes))
			#print(abs(s_h_form - s_w))
	end = time.time()

	print(f"gensim: {end - start}")
