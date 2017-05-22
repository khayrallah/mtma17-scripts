#/usr/bin/env python

"""
Get parallel sentences from a training set
-- Sentence length <= 10 words
-- Unique
-- Sentences should be non-empty
"""

import codecs
import argparse
import hashlib
from itertools import izip

MAX_SENT_LEN=10

#TODO: will currently write to the location where the source and the target files live
parser = argparse.ArgumentParser(description='Sample parallel sentences')
parser.add_argument('-s', '--source', required=True, help='Source corpus')
parser.add_argument('-t', '--target', required=True, help='Target corpus')
parser.add_argument('-n', type=int, required=True, help="Number of samples")
args = parser.parse_args()

s_hashset = set()
t_hashset = set()
n_sampled = 0

print "Sampling a max of", args.n, "lines"

out_source = codecs.open(args.source + "." + str(args.n), 'w', encoding='utf8')
out_target = codecs.open(args.target + "." + str(args.n), 'w', encoding='utf8')

with codecs.open(args.source, encoding='utf8') as sf, \
    codecs.open(args.target, encoding='utf8') as tf:

    for s, t in izip(sf, tf):
        if len(s.split()) > MAX_SENT_LEN or len(t.split()) > MAX_SENT_LEN:
            continue
        if len(s.strip()) == 0 or len(t.strip()) == 0:
            continue
        s_hash = hashlib.sha1(s.encode('utf8')).digest()
        t_hash = hashlib.sha1(t.encode('utf8')).digest()
        if s_hash in s_hashset or t_hash in t_hashset:
            continue
        n_sampled += 1
        s_hashset.update({s_hash})
        t_hashset.update({t_hash})

        out_source.write(s)
        out_target.write(t)

        if n_sampled >= args.n:
            break

out_source.close()
out_target.close()
