from __future__ import print_function
from __future__ import division

from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy
from lshash import LSHash
import time

start = time.time()
lsh = LSHash(8, 300)
sample_word_embeds = []
for i in tqdm(xrange(20000)):
    word_embed = numpy.random.rand(300)
    lsh.index(word_embed)

    if i % 500 == 0:
        sample_word_embeds.append(word_embed)

print("Indexing takes {} seconds".format(time.time() - start))

start = time.time()
for word_embed in sample_word_embeds:
    print('-' * 80)
    results = lsh.query(word_embed, num_results=None, distance_func='cosine')
    print("Num result: {}".format(len(results)))
    print('Nearest neighbor cosine distance:')
    print("    {} | {}".format(results[1][1], cosine(results[1][0], word_embed)))

print('Query takes average {} seconds'.format((time.time() - start) / len(sample_word_embeds)))