import numpy as np
import re
import itertools
import collections
import os
import nltk
import codecs



def prepare_data(path, vocabsize):
    count = [['<unk>', -1]]
    outdir = "data/gutenberg_clean/"
    # Load data from files
    with open((os.path.join(path, "train.txt")), "r",  encoding = "ISO-8859-1") as f:
        words = [word for line in f for word in line.split()]
        count.extend(collections.Counter(words).most_common(vocabsize - 1))
        count = dict(count)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for filename in os.listdir(path):
        text = open((os.path.join(path, filename)), "r",  encoding = "ISO-8859-1").readlines()
        outfile = codecs.open((outdir + filename), "w", "utf-8")
        for line in text:
            lst = line.split()
            no_unks = [w if w in count.keys() else '<unk>' for w in lst]
            no_nums = [w if not w.isdigit() else 'N' for w in no_unks]
            outfile.write(' '.join(no_nums) + '\n')

