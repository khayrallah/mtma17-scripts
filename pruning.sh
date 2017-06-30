# Phrase table / lexicon pruning w/ significance based Moses code and custom n-best
# sjmielke@jhu.edu
# edits to make it more generalized - huda@jhu.edu


#usage: prunning.sh moses-working-directory  run-number prunning-directory src trg

MOSES_DIR=$1
RUN=$2
DIR=$3
SRC=$4
TRG=$5

cd $DIR

# Get phrase tables
ln -s $MOSES_DIR/model/phrase-table.${RUN}.gz  phrase-table.gz  
PHRASE_TABLE=phrase-table.gz  


# Get lexicons (swapping the f2e so its actually f2e!)
dict=$MOSES_DIR/model/lex.${RUN}.f2e
for col in 1 2 3; do
	cut -d ' ' -f $col $dict > tmp.flip$col
done

SWAP_DICT=lexicon.f2e.swap.gz
paste -d ' ' tmp.flip{2,1,3} | sed 's/ / ||| /g;s/$/ ||| ||| /' | gzip > $SWAP_DICT



#creates suffix array of source and target sides
#https://github.com/moses-smt/mosesdecoder/blob/master/contrib/sigtest-filter/README.txt
# Initialize filtering tool



for lang in $SRC $TRG; do
	ln -s $MOSES_DIR/corpus/*.clean.$RUN.$lang train.$lang
	/home/smielke/salm/Bin/Linux/Index/IndexSA.O64 train.$lang
done

# Prune both w/ significance-based pruning and then manual pruning with n best
n=5


for FILE in $SWAP_DICT  $PHRASE_TABLE ; do
	zcat $FILE | /home/smielke/mosesdecoder/contrib/sigtest-filter/filter-pt -e train.$TRG -f train.$SRC -l a+e 2> /dev/null | LC_COLLATE=C sort > $FILE.pruned-a+e
	python3<<EOF

newlines = []
with open("$FILE.pruned-a+e", 'r', encoding='utf-8') as f:
        current_source = None
        current_group = []
        for line in f.read().splitlines():
                l = line.split(" ||| ")
                source = l[0]
                target = l[1]
                score = l[2]
                if source == current_source:
                        current_group.append((score, line))
                elif source != current_source:
                        newlines += [l for (s,l) in sorted(current_group, reverse=True)[0:$n]]
                        current_group = []
                        current_source = source
        
        newlines += [l for (s,l) in sorted(current_group, reverse=True)[0:$n]]
with open("$FILE.pruned-a+e-n$n", 'w', encoding='utf-8') as f:
        print('\n'.join(newlines), file=f)
EOF
#get 1st and 3rd colums to turn these into taining data

cat $FILE.pruned-a+e-n$n | awk -F" ||| " '{ print $1 }' > $FILE.pruned-a+e-n$n.$SRC
cat $FILE.pruned-a+e-n$n | awk -F" ||| " '{ print $3 }' > $FILE.pruned-a+e-n$n.$TRG


done
	
rm tmp*

