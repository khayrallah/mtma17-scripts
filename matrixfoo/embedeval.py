from lshash.lshash import LSHash
from bitarray import bitarray
from math import cos
import numpy as np
from scipy.spatial.distance import cosine

import time
import sys

vocab_size = 10000

import pickle
with open('/mnt/hylia/embed_wiki_de_2M_52D/vocabulary.pickle', 'rb') as f:
	o = pickle.load(f)
	source_words = set([w for (s, w) in sorted([(s, w) for (w,s) in o.items()])][0:vocab_size])
print("Got vocab.", file = sys.stderr)

model = {}
i = 0
with open("/mnt/hylia/embed_wiki_de_2M_52D/embedding_file") as f:
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
print(len(model), vocab_size)
print("Got embeddings.", file = sys.stderr)

planes = 1024
lsh = LSHash(planes, dims)
source_hashes = [bitarray(lsh._hash(lsh.uniform_planes[0], model[w])) for w in source_words]
print("Got hashes.", file = sys.stderr)

def simi(w1, w2):
	h1 = bitarray(lsh._hash(lsh.uniform_planes[0], model[w1]))
	h2 = bitarray(lsh._hash(lsh.uniform_planes[0], model[w2]))
	return cos(3.141592653589 * float((h1 ^ h2).count()) / float(planes))

print(simi("weg", "straße"))
print(simi("weg", "pfad"))
print(simi("weg", "fluss"))
print(simi("weg", "abstrakt"))

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
